#!/usr/bin/env python3
import sys
import subprocess
import pathlib

# --- CONFIG -------------------------------------------------------

# All outputs must:
# - have the same duration as the input
# - be 48 kHz (either via -ar or via filter chain)
#
# For the low-pass variant we enforce the 8 kHz ceiling by:
#   aresample=16000,aresample=48000
# which guarantees no energy above 8 kHz.

PROFILES = [
    {
        "name": "mp3_120k",
        "codec": "libmp3lame",
        "bitrate": "120k",
        "ext": "mp3",
        "sample_rate": 48000,
        "filter": None,
        "extra_args": [],
    },
    {
        "name": "aac_120k",
        "codec": "aac",
        "bitrate": "120k",
        "ext": "m4a",  # AAC in MP4 container
        "sample_rate": 48000,
        "filter": None,
        "extra_args": ["-movflags", "+faststart"],
    },
    {
        "name": "opus_30k",
        "codec": "libopus",
        "bitrate": "30k",
        "ext": "opus",
        "sample_rate": 48000,
        "filter": None,
        "extra_args": [],
    },
    {
        "name": "aac_320k",
        "codec": "aac",
        "bitrate": "320k",
        "ext": "m4a",
        "sample_rate": 48000,
        "filter": None,
        "extra_args": ["-movflags", "+faststart"],
    },
    {
        "name": "flac",
        "codec": "flac",
        "bitrate": None,  # lossless
        "ext": "flac",
        "sample_rate": 48000,
        "filter": None,
        "extra_args": [],
    },
    # Hard 8 kHz low-pass, output at 48 kHz
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


def transcode_file(input_path: pathlib.Path):
    if not input_path.is_file():
        print(f"[WARN] Not a file: {input_path}")
        return

    duration = get_duration_seconds(input_path)
    duration_str = f"{duration:.6f}"

    out_dir = input_path.parent / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem

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

        # Sample rate if defined for this profile
        if profile.get("sample_rate") is not None:
            cmd += ["-ar", str(profile["sample_rate"])]

        # Filter chain (for low-pass profile this does the 16k -> 48k trick)
        if profile.get("filter"):
            cmd += ["-af", profile["filter"]]

        # Codec
        cmd += ["-c:a", profile["codec"]]

        # Bitrate where applicable
        if profile.get("bitrate") is not None:
            cmd += ["-b:a", profile["bitrate"]]

        # Any profile-specific extras
        cmd += profile.get("extra_args", [])

        cmd.append(str(out_path))

        run_ffmpeg(cmd)


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
