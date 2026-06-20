"""
train.py
========
Training pipeline for the HCTF (Hierarchical Cross-Transformer Fusion) model.

Workflow
--------
1. Load dual-modal HDF5 feature files (PATCH + MGAF) and label array.
2. Stratified train/test split (default 75/25).
3. Build the training-stage HCTF model (build_dual_mgaf_transformer).
4. Train with Adam optimiser + categorical cross-entropy.
5. ModelCheckpoint saves *only the best validation-accuracy weights*
   to a .weights.h5 file compatible with build_explainable_hctf.
6. Export training curves, confusion matrix, and test metrics to CSV/TIFF.
7. Validate weight transfer by loading saved weights into the XAI engine
   and running a sample forward pass to confirm all four output tensors.

Usage
-----
  python train.py \\
      --patch  X_patch.h5 \\
      --mgaf   X_mgaf.h5  \\
      --labels y_labels.h5 \\
      --epochs 50 \\
      --output-dir training_results

Expected HDF5 file shapes
-------------------------
  X_patch.h5  : (N, 60, 10)       -- PCA-guided spatial-spectral patch features
  X_mgaf.h5   : (N, 60, 10, 10)   -- MGAF structural correlation matrices
  y_labels.h5 : (N,)               -- integer labels {0: Control, 1: AC, 2: NC}
"""

import os
import time
import argparse
import itertools

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server / headless environments
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.utils import to_categorical
from keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from model_def import build_dual_mgaf_transformer, build_explainable_hctf


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_h5(file_path: str) -> np.ndarray:
    """
    Load a feature or label array from an HDF5 file.

    Expects the file to contain a single top-level dataset.  If the leading
    dimension is 1 (i.e. the array was saved with an extra batch axis), it is
    squeezed automatically.

    Parameters
    ----------
    file_path : str
        Path to the .h5 file.

    Returns
    -------
    np.ndarray
    """
    print(f"  [load] Reading: {file_path}")
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        data = f[keys[0]][()]
    # Handle optional leading singleton dimension
    if data.ndim > 1 and data.shape[0] == 1:
        data = np.squeeze(data, axis=0)
    return data


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_training_results(
    history,
    model,
    x_test,
    y_test: np.ndarray,
    output_dir: str,
    runtime: float,
    num_classes: int = 3,
) -> None:
    """
    Export training artifacts to *output_dir*:
      - training_results.csv  : per-epoch loss and accuracy
      - test_results.csv      : final test metrics (accuracy, precision, recall, F1)
      - loss_curve.tif        : 300 DPI loss curve (TIFF)
      - accuracy_curve.tif    : 300 DPI accuracy curve (TIFF)
      - confusion_matrix.tif  : 300 DPI confusion matrix (TIFF)

    Parameters
    ----------
    history      : Keras History object returned by model.fit()
    model        : compiled Keras model (training variant)
    x_test       : list [X_patch_test, X_mgaf_test]
    y_test       : integer label array for the test split
    output_dir   : destination directory (created if absent)
    runtime      : total training time in seconds
    num_classes  : number of output classes
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams["font.family"] = "Times New Roman"
    print(f"  [save] Writing results to '{output_dir}' ...")

    # Infer history key names (TF version compatibility)
    h = history.history
    acc_key = "acc" if "acc" in h else "accuracy"
    val_acc_key = "val_acc" if "val_acc" in h else "val_accuracy"

    # --- Training history CSV ---
    pd.DataFrame({
        "loss": h["loss"], "val_loss": h["val_loss"],
        "acc":  h[acc_key], "val_acc": h[val_acc_key],
    }).to_csv(os.path.join(output_dir, "training_results.csv"), index=False)

    # --- Test metrics ---
    y_test_cat = to_categorical(y_test, num_classes=num_classes)
    test_loss, test_acc, test_prec, test_rec = model.evaluate(
        x_test, y_test_cat, verbose=0
    )
    test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec + 1e-8)
    pd.DataFrame([{
        "test_loss": test_loss, "test_accuracy": test_acc,
        "test_precision": test_prec, "test_recall": test_rec,
        "test_f1_score": test_f1, "runtime_s": runtime,
    }]).to_csv(os.path.join(output_dir, "test_results.csv"), index=False)

    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(h["loss"], label="Training loss")
    ax.plot(h["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curve.tif"),
                dpi=300, format="tiff", bbox_inches="tight")
    plt.close(fig)

    # --- Accuracy curve ---
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(h[acc_key], label="Training accuracy")
    ax.plot(h[val_acc_key], label="Validation accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_curve.tif"),
                dpi=300, format="tiff", bbox_inches="tight")
    plt.close(fig)

    # --- Confusion matrix ---
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    class_names = ["Control", "AC", "NC"]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_xticklabels(class_names, rotation=30)
    ax.set_yticks(ticks); ax.set_yticklabels(class_names)
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], ha="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.tif"),
                dpi=300, format="tiff", bbox_inches="tight")
    plt.close(fig)

    print(f"  [save] Done.  Test accuracy: {test_acc:.4f}  "
          f"F1: {test_f1:.4f}  Runtime: {runtime:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train the HCTF model on dual-modal hyperspectral features."
    )
    p.add_argument("--patch",  default="X_patch.h5",
                   help="Path to PATCH feature HDF5 file  [shape: N×60×10]")
    p.add_argument("--mgaf",   default="X_mgaf.h5",
                   help="Path to MGAF feature HDF5 file   [shape: N×60×10×10]")
    p.add_argument("--labels", default="y_labels.h5",
                   help="Path to integer label HDF5 file   [shape: N]")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=128)
    p.add_argument("--test-size",   type=float, default=0.25,
                   help="Fraction of data reserved for testing (default: 0.25)")
    p.add_argument("--embed-dim",   type=int,   default=64)
    p.add_argument("--num-heads",   type=int,   default=4)
    p.add_argument("--ff-dim",      type=int,   default=128)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--output-dir",  default="training_results",
                   help="Directory for exported curves, metrics, and figures")
    p.add_argument("--weights-path", default="hctf_best_deployment.weights.h5",
                   help="Destination path for the best-checkpoint weight file")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for stratified train/test split")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 64)
    print("  HCTF Training Pipeline")
    print("=" * 64)

    # --- 1. Load data ---
    print("[1/7] Loading feature files ...")
    try:
        X_patch = load_h5(args.patch)    # (N, 60, 10)
        X_mgaf  = load_h5(args.mgaf)     # (N, 60, 10, 10)
        y       = load_h5(args.labels)   # (N,)
    except Exception as e:
        raise SystemExit(f"[ERROR] Failed to load data: {e}")

    # Replace any NaN / Inf values in MGAF matrices
    X_mgaf = np.nan_to_num(X_mgaf, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Loaded {len(y)} samples — "
          f"PATCH {X_patch.shape}, MGAF {X_mgaf.shape}, labels {y.shape}")

    # --- 2. Stratified split ---
    print("[2/7] Stratified train/test split ...")
    X1_tr, X1_te, X2_tr, X2_te, y_tr, y_te = train_test_split(
        X_patch, X_mgaf, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    num_classes = int(y.max()) + 1
    y_tr_cat = to_categorical(y_tr, num_classes=num_classes)
    y_te_cat = to_categorical(y_te, num_classes=num_classes)
    print(f"  Training: {len(y_tr)}  |  Test: {len(y_te)}  |  "
          f"Classes: {num_classes}")

    # --- 3. Build model ---
    print("[3/7] Building HCTF training model ...")
    model = build_dual_mgaf_transformer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_classes=num_classes,
        dropout_rate=args.dropout,
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["acc", Precision(name="precision"), Recall(name="recall")],
    )
    model.summary(line_length=80)

    # --- 4. Callbacks ---
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.weights_path,
        monitor="val_acc",
        mode="max",
        save_best_only=True,
        save_weights_only=True,   # critical: enables loading into XAI model
        verbose=1,
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=15, restore_best_weights=False, verbose=1
    )

    # --- 5. Training ---
    print(f"[4/7] Training for up to {args.epochs} epochs "
          f"(batch_size={args.batch_size}) ...")
    t0 = time.time()
    history = model.fit(
        [X1_tr, X2_tr], y_tr_cat,
        validation_data=([X1_te, X2_te], y_te_cat),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[checkpoint, early_stop],
        verbose=1,
    )
    runtime = time.time() - t0
    print(f"[5/7] Training complete in {runtime:.1f}s.")

    # --- 6. Export results ---
    print("[6/7] Exporting training artifacts ...")
    save_training_results(
        history, model, [X1_te, X2_te], y_te,
        output_dir=args.output_dir,
        runtime=runtime,
        num_classes=num_classes,
    )

    # --- 7. Validate XAI weight transfer ---
    print("[7/7] Validating weight transfer to XAI deployment engine ...")
    xai_model = build_explainable_hctf(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_classes=num_classes,
        dropout_rate=args.dropout,
    )
    try:
        xai_model.load_weights(args.weights_path)
        out = xai_model.predict([X1_te[:2], X2_te[:2]], verbose=0)
        print("  [OK] Weights loaded successfully into XAI engine.")
        print(f"       cls_output        : {out['cls_output'].shape}")
        print(f"       cross_attention   : {out['cross_attention'].shape}")
        print(f"       self_attention    : {out['self_attention'].shape}")
        print(f"       mgaf_token_scores : {out['mgaf_token_scores'].shape}")
    except Exception as e:
        print(f"  [WARNING] XAI weight transfer check failed: {e}")

    print("=" * 64)
    print(f"  Best weights : {args.weights_path}")
    print(f"  Results dir  : {args.output_dir}/")
    print("=" * 64)


if __name__ == "__main__":
    main()