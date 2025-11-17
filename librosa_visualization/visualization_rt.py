#!/usr/bin/env python3
import threading
import numpy as np
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
AUDIO_FILE = "Rache.wav"  # <-- put your file here
BLOCKSIZE = 1024                           # audio block size (frames)
DB_FLOOR = -100.0                          # dB floor for plotting
# ----------------------------------------


def main():
    # 1) Load audio (mono) with librosa
    print(f"Loading {AUDIO_FILE} ...")
    y, sr = librosa.load(AUDIO_FILE, sr=None, mono=True)
    print(f"Loaded: {len(y)} samples @ {sr} Hz")

    # Normalise slightly for safety
    peak = np.max(np.abs(y))
    if peak > 0:
        y[:] = y / peak * 0.9

    # 2) Prepare FFT window & frequency axis
    window = np.hanning(BLOCKSIZE).astype(np.float32)
    freqs = np.fft.rfftfreq(BLOCKSIZE, d=1.0 / sr)

    # Shared state between callback and GUI
    spec_lock = threading.Lock()
    current_spectrum_db = np.full(freqs.shape, DB_FLOOR, dtype=np.float32)
    pos = 0  # playback position in samples

    # 3) Define audio callback
    def audio_callback(outdata, frames, time_info, status):
        nonlocal pos, current_spectrum_db

        if status:
            print(status)

        # Slice the next chunk
        chunk = y[pos:pos + frames]

        # If we reached the end, pad with zeros and stop
        if len(chunk) < frames:
            outdata[:] = 0.0
            if len(chunk) > 0:
                outdata[:len(chunk), 0] = chunk
                # compute spectrum on last partial block (zero-padded)
                padded = np.zeros(frames, dtype=np.float32)
                padded[:len(chunk)] = chunk
                frame = padded
            else:
                frame = np.zeros(frames, dtype=np.float32)

            # FFT for visualization
            windowed = frame * window
            fft = np.fft.rfft(windowed)
            mag = np.abs(fft)
            mag_db = 20.0 * np.log10(mag + 1e-12)
            mag_db = np.clip(mag_db, DB_FLOOR, 0.0)

            with spec_lock:
                current_spectrum_db = mag_db.astype(np.float32)

            raise sd.CallbackStop()

        # Normal case: full block available
        outdata[:, 0] = chunk  # mono -> first channel
        # If you want stereo-dual-mono: outdata[:, 1] = chunk (and set channels=2)

        frame = chunk.astype(np.float32)
        if len(frame) < BLOCKSIZE:
            # Shouldn't really happen here, but be safe
            padded = np.zeros(frames, dtype=np.float32)
            padded[:len(frame)] = frame
            frame = padded

        windowed = frame * window
        fft = np.fft.rfft(windowed)
        mag = np.abs(fft) / (BLOCKSIZE / 2.0)
        mag_db = 20.0 * np.log10(mag + 1e-12)

        with spec_lock:
            current_spectrum_db = mag_db.astype(np.float32)

        pos += frames

    # 4) Set up matplotlib in interactive mode
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot(freqs, current_spectrum_db)
    ax.set_xlim(0, sr / 2)
    ax.set_ylim(DB_FLOOR, 0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title("Real-time spectrum (callback-based)")
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # 5) Start audio stream with callback
    print("Starting stream...")
    with sd.OutputStream(
        samplerate=sr,
        blocksize=BLOCKSIZE,
        channels=1,      # change to 2 and duplicate in callback if you want stereo dual-mono
        dtype="float32",
        callback=audio_callback,
    ) as stream:

        # 6) GUI update loop while stream is active
        while stream.active:
            with spec_lock:
                spec = current_spectrum_db.copy()

            line.set_ydata(spec)
            ax.set_title("Real-time spectrum (callback-based)")
            fig.canvas.draw()
            fig.canvas.flush_events()

            plt.pause(0.01)  # ~100 FPS max, adjust if needed

    print("Playback finished.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
