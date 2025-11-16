#!/usr/bin/env python3
import os
import glob
import argparse

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# ---------------- CONFIG ----------------
TRAIN_DIR = "Training"
TEST_DIR  = "Testing"

SR         = 22050   # target sample rate
N_MELS     = 64      # mel bands
N_FFT      = 2048
HOP_LENGTH = 512
# ----------------------------------------


def discover_classes(train_dir):
    """
    Discover instrument classes from subdirectories of TRAIN_DIR.
    E.g. Training/sax, Training/vio -> ['sax', 'vio']
    """
    classes = [
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ]
    classes.sort()
    if not classes:
        raise RuntimeError(f"No class subdirectories found under {train_dir}")
    return classes


def extract_features(path):
    """
    Log-mel spectrogram -> mean + std over time.
    Produces a fixed-length feature vector from an arbitrary-length wav.
    """
    y, sr = librosa.load(path, sr=SR, mono=True)

    if y.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    mean = S_db.mean(axis=1)
    std  = S_db.std(axis=1)

    return np.hstack([mean, std])  # shape: (2 * N_MELS,)


def build_training_set(train_dir):
    """
    Walk Training/<class>/*.wav and build X, y.
    """
    class_names = discover_classes(train_dir)
    print("Discovered classes:", class_names)

    X = []
    y = []

    for label_idx, cls in enumerate(class_names):
        pattern = os.path.join(train_dir, cls, "*.wav")
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"[WARN] No wav files found for class '{cls}'")
            continue

        print(f"[INFO] Class '{cls}': {len(files)} file(s)")

        for f in files:
            try:
                feat = extract_features(f)
                X.append(feat)
                y.append(label_idx)
            except Exception as e:
                print(f"[ERROR] Failed on {f}: {e}")

    if not X:
        raise RuntimeError("No training data assembled; check your paths.")

    X = np.array(X)
    y = np.array(y)
    print("Training feature matrix shape:", X.shape)
    return X, y, class_names


def train_classifier(X, y):
    """
    Tiny dataset => keep it simple; this will overfit but is fine for a demo.
    """
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X, y)
    return clf


def predict_on_testing(clf, class_names, test_dir):
    """
    Run predictions on all Testing/*.wav and print results.
    """
    pattern = os.path.join(test_dir, "*.wav")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[WARN] No wav files found in {test_dir}")
        return

    print(f"\n=== Inference on {test_dir}/ ===")
    for f in files:
        try:
            feat = extract_features(f).reshape(1, -1)
            pred_idx = clf.predict(feat)[0]
            proba = clf.predict_proba(feat)[0]

            predicted_label = class_names[pred_idx]
            proba_dict = {cls: float(p) for cls, p in zip(class_names, proba)}

            print(f"\nFile: {os.path.basename(f)}")
            print(f"Predicted class: {predicted_label}")
            print(f"Probabilities: {proba_dict}")
        except Exception as e:
            print(f"[ERROR] Failed on {f}: {e}")


def visualize_file_features(path):
    """
    Show:
    - waveform
    - mel-spectrogram
    - MFCCs
    - final aggregated feature vector used for classification
    """
    print(f"\n[VIS] Visualizing features for: {path}")

    y, sr = librosa.load(path, sr=SR, mono=True)

    # ---- Waveform ----
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform: {os.path.basename(path)}")
    plt.tight_layout()

    # ---- Mel-spectrogram (same config as extract_features) ----
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-mel spectrogram: {os.path.basename(path)}")
    plt.tight_layout()

    # ---- MFCCs ----
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mfcc,
        x_axis="time",
        sr=sr,
        hop_length=HOP_LENGTH
    )
    plt.colorbar()
    plt.title(f"MFCCs: {os.path.basename(path)}")
    plt.tight_layout()

    # ---- Final feature vector (mean+std) ----
    feat = extract_features(path)
    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(len(feat)), feat)
    plt.title(f"Aggregated feature vector (mean+std): {os.path.basename(path)}")
    plt.xlabel("Feature index")
    plt.ylabel("Value")
    plt.tight_layout()

    plt.show()


def visualize_dataset_space(X, y, class_names):
    """
    Project feature vectors to 2D with PCA and scatter-plot them.
    With your tiny dataset it’s mostly illustrative, but the code scales.
    """
    if X.shape[0] < 2:
        print("[INFO] Not enough samples to visualize dataset space.")
        return

    print("[VIS] Visualizing dataset feature space (PCA 2D)")

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))
    for idx, cls in enumerate(class_names):
        mask = (y == idx)
        if not np.any(mask):
            continue
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            label=cls,
            alpha=0.8
        )

    plt.title("Feature space (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def find_first_training_file(train_dir, class_names):
    """
    Get first available training wav to visualize if user
    didn't specify any file.
    """
    for cls in class_names:
        pattern = os.path.join(train_dir, cls, "*.wav")
        files = sorted(glob.glob(pattern))
        if files:
            return files[0]
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Toy instrument classifier using librosa + RandomForest."
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable feature visualizations (PCA + one file)."
    )
    parser.add_argument(
        "--viz-file",
        type=str,
        default=None,
        help="Specific audio file to visualize when --viz is set. "
             "If omitted, first training file is used."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Assemble training set
    X, y, class_names = build_training_set(TRAIN_DIR)

    if len(y) < 2:
        print("[WARN] Less than 2 training samples; classifier will be weak.")

    # 2) Train model
    clf = train_classifier(X, y)

    # 3) Predict on Testing/
    predict_on_testing(clf, class_names, TEST_DIR)

    # 4) Optional visualizations
    if args.viz:
        visualize_dataset_space(X, y, class_names)

        if args.viz_file is not None:
            viz_path = args.viz_file
        else:
            viz_path = find_first_training_file(TRAIN_DIR, class_names)

        if viz_path is None:
            print("[WARN] No training files found to visualize.")
        else:
            visualize_file_features(viz_path)


if __name__ == "__main__":
    main()
