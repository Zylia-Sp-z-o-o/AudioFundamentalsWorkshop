import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

filename = "Rache.wav"
y, sr = librosa.load(filename, sr=None)

# Short-time Fourier transform
S = librosa.stft(y, n_fft=2048, hop_length=512)
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db,
                         sr=sr,
                         hop_length=512,
                         x_axis='time',
                         y_axis='log')  # or 'linear'
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (STFT magnitude in dB)")
plt.tight_layout()
plt.show()
