#!/usr/bin/env python3
import sys
import subprocess
import pathlib
import numpy as np

# --- CONFIG -------------------------------------------------------

# All outputs must:
# - have the same duration
# - be 48 kHz (either via -ar or via filter chain)
# - be dual-mono: same signal on both channels
#
# SNR is computed vs the processed reference:
#   ref_48k_dualmono (48 kHz, dual mono, PCM WAV).
#
# For the low-pass variant we enforce the 8 kHz ceiling by:
#   aresample=16000,aresample=48000
# then duplicate to stereo.

PROFILES = [
    # MUSHRA reference: 48k, dual mono, PCM WAV
    {
        "name": "ref_48k_dualmono",
        "codec": "pcm_s24le",
        "bitrate": None,
        "ext": "wav",
        "sample_rate": 48000,
        "filter": None,
        "extra_args": [],
    },

    {
        "name": "mp3_96k",
        "codec": "libmp3lame",
        "bitrate": "96k",
        "ext": "mp3",
        "sample_rate": 48000,
        "filter": None,
        "extra_args": [],
    },
    {
        "name": "aac_64k",
        "codec": "aac",
        "bitrate": "64k",
        "ext": "m4a",           # AAC in MP4 container
        "sample_rate": 48000,
        "filter": None,
        "extra_args": ["-movflags", "+faststart"],
    },
    {
        "name": "opus_24k",
        "codec": "libopus",
        "bitrate": "24k",
        "ext": "opus",
        "sample_rate": 48000,
        "filter": None,
        "extra_args": [],
    },
    {
        "name": "aac_32k",
        "codec": "aac",
        "bitrate": "32k",
        "ext": "m4a",
        "sample_rate": 48000,
        "filter": None,
        "extra_args": ["-movflags", "+faststart"],
    },
    {
        "name": "flac",
        "codec": "flac",
        "bitrate": None,        # lossless
        "ext": "flac",
        "sample_rate": 48000,
        "filter": None,
        "extra_args": [],
    },
    # Hard 8 kHz low-pass, output at 48 kHz, dual mono
    {
        "name": "lp8k_wav",
        "codec": "pcm_s16le",          # WAV
        "bitrate": None,
        "ext": "wav",
        "sample_rate": None,           # handled by filter chain
        # downsample to 16 kHz (Nyquist=8 kHz), then back to 48 kHz
        "filter": "aresample=16000,aresample=48000",
        "extra_args": [],
    },
]

OUTPUT_SUBDIR = "compressed"
TARGET_SNR_SR = 48000   # sample rate used for SNR calculations (mono)
# -----------------------------------------------------------------


def run_ffmpeg(cmd):
    print(" ".join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] ffmpeg failed with return code {result.returncode}")


def get_duration_seconds(path: pathlib.Path) -> float:
    """
    Use ffprobe to get exact input duration in seconds.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr}")

    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"Could not parse duration for {path}: {result.stdout}")


def decode_to_numpy(path: pathlib.Path, target_sr: int = TARGET_SNR_SR) -> np.ndarray:
    """
    Decode any audio file to mono float32 at target_sr using ffmpeg.
    Returns a 1D np.ndarray.
    """
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(path),
        "-vn",
        "-ac", "1",               # mono for SNR reference/comparison
        "-ar", str(target_sr),
        "-f", "f32le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed for {path}: {result.stderr.decode(errors='ignore')}"
        )
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio


def compute_snr(ref: np.ndarray, test: np.ndarray) -> float:
    """
    Compute SNR in dB between reference and test signals.
    Both must be 1D, same sample rate. Length is aligned to min length.
    """
    n = min(len(ref), len(test))
    if n == 0:
        raise ValueError("Zero-length signals for SNR computation.")
    ref = ref[:n]
    test = test[:n]
    noise = ref - test

    sig_power = np.mean(ref ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power <= 0:
        return float("inf")

    snr = 10.0 * np.log10(sig_power / noise_power)
    return snr


def transcode_file(input_path: pathlib.Path):
    if not input_path.is_file():
        print(f"[WARN] Not a file: {input_path}")
        return

    duration = get_duration_seconds(input_path)
    duration_str = f"{duration:.6f}"

    out_dir = input_path.parent / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem

    produced_outputs = []  # list of (profile_name, out_path)

    for profile in PROFILES:
        out_name = f"{base_name}_{profile['name']}.{profile['ext']}"
        out_path = out_dir / out_name

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vn",                 # audio only
            "-t", duration_str,    # lock duration
        ]

        # Target sample rate (if explicitly configured for this profile)
        if profile.get("sample_rate") is not None:
            cmd += ["-ar", str(profile["sample_rate"])]

        # Build filter chain:
        #   - profile-specific base filter (e.g., resample low-pass)
        #   - dual mono: pan=stereo|c0=c0|c1=c0
        base_filter = profile.get("filter")
        dual_mono_filter = "pan=stereo|c0=c0|c1=c0"
        if base_filter:
            full_filter = f"{base_filter},{dual_mono_filter}"
        else:
            full_filter = dual_mono_filter

        cmd += ["-af", full_filter]

        # Force 2 output channels (dual mono)
        cmd += ["-ac", "2"]

        # Codec
        cmd += ["-c:a", profile["codec"]]

        # Bitrate where applicable
        if profile.get("bitrate") is not None:
            cmd += ["-b:a", profile["bitrate"]]

        # Any profile-specific extras
        cmd += profile.get("extra_args", [])

        cmd.append(str(out_path))
        print (cmd)
        run_ffmpeg(cmd)
        produced_outputs.append((profile["name"], out_path))

    # --- SNR evaluation -------------------------------------------------
    # Use the *processed reference* as SNR baseline, not the raw original.
    ref_path = None
    for prof_name, out_path in produced_outputs:
        if prof_name == "ref_48k_dualmono":
            ref_path = out_path
            break

    if ref_path is None:
        print("[ERROR] ref_48k_dualmono not found among outputs; cannot compute SNR.")
        return

    try:
        ref_audio = decode_to_numpy(ref_path, target_sr=TARGET_SNR_SR)
    except Exception as e:
        print(f"[ERROR] Could not decode processed reference for SNR: {e}")
        return

    print("\nSNR vs ref_48k_dualmono ({} Hz, mono):".format(TARGET_SNR_SR))
    for prof_name, out_path in produced_outputs:
        try:
            test_audio = decode_to_numpy(out_path, target_sr=TARGET_SNR_SR)
            snr = compute_snr(ref_audio, test_audio)
            if np.isinf(snr):
                snr_str = "inf"
            else:
                snr_str = f"{snr:.2f} dB"
            print(f"  {prof_name:16s}: {snr_str}")
        except Exception as e:
            print(f"  {prof_name:16s}: SNR failed ({e})")


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_transcode.py file1.wav [file2.wav ...]")
        sys.exit(1)

    input_files = [pathlib.Path(p) for p in sys.argv[1:]]

    for f in input_files:
        print(f"\n=== Processing: {f} ===")
        transcode_file(f)


if __name__ == "__main__":
    main()
