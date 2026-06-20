"""
app.py
======
SpectraX-Inspect FastAPI inference server.

Implements the asynchronous, high-throughput SaaS backend described in
manuscript Section 3.7.1.  Key design principles:

  1. Batch pre-computation and in-memory indexing
     On receipt of a dual-modal HDF5 upload, the server runs GPU-accelerated
     batch inference (batch_size=128) across the complete dataset in one pass.
     All per-sample outputs (probabilities + MGAF token scores) are cached in
     RESULT_DB.  Subsequent sample queries are resolved by O(1) index lookup —
     no repeated model calls.

  2. Explainability-by-architecture
     The deployment engine (build_explainable_hctf) mirrors the exact weight
     topology of the training model but additionally exposes cross-modal
     attention tensors and MGAF spectral gate scores in every forward pass.
     This is structurally different from post-hoc approximation methods.

  3. Dual-resolution visualization support
     /api/v1/analyze-batch  — returns macro population statistics and
                              wavelength-resolved biomarker activation curves
                              for dashboard-level visualisation.
     /api/v1/sample/{id}   — returns per-kernel probabilities and 60-element
                              MGAF token score vector for spatial heatmap
                              rendering in the micro-explorer panel.

Usage
-----
  uvicorn app:app --host 0.0.0.0 --port 8000

or run directly:
  python app.py [--weights hctf_best_deployment.weights.h5] [--port 8000]
"""

import os
import argparse
import tempfile

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — prevent GUI conflicts
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from model_def import build_explainable_hctf


# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SpectraX-Inspect advanced expert system",
    description=(
        "High-throughput hyperspectral postharvest quality control via the "
        "HCTF (Hierarchical Cross-Transformer Fusion) framework.  "
        "Provides macro population analytics and per-kernel pathological traceability."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Physical wavelengths of the 10 PCA-selected biomarker bands (nm)
# Corresponds to the 2159–2216 nm narrow-band window (manuscript Section 3.5.3)
WAVELENGTHS = [
    "2159.0", "2165.3", "2171.7", "2178.0", "2184.4",
    "2190.7", "2197.1", "2203.4", "2209.8", "2216.0",
]
CLASS_NAMES = ["Control", "AC", "NC"]
CLASS_COLORS = ["#15803d", "#b45309", "#b91c1c"]   # green, orange, red

# In-memory result store (batch pre-computation cache)
RESULT_DB: dict = {
    "total_samples": 0,
    "probs": None,   # ndarray (N, 3)
    "scores": None,  # ndarray (N, 60)
}


# ---------------------------------------------------------------------------
# Model initialisation
# ---------------------------------------------------------------------------

def _parse_weights_path() -> str:
    """Extract --weights argument without interfering with uvicorn's own args."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--weights", default="hctf_best_deployment.weights.h5")
    args, _ = parser.parse_known_args()
    return args.weights


WEIGHTS_PATH = _parse_weights_path()
print("[SYSTEM] Initialising HCTF dual-modal inference engine ...")
_model = build_explainable_hctf(embed_dim=64, num_heads=4, ff_dim=128)
try:
    _model.load_weights(WEIGHTS_PATH)
    print(f"[SYSTEM] Weights loaded from '{WEIGHTS_PATH}'.  Ready.")
except Exception as e:
    print(f"[WARNING] Could not load weights from '{WEIGHTS_PATH}': {e}")
    print("          Start the server only after running train.py.")


# ---------------------------------------------------------------------------
# SCI-grade TIFF figure export (triggered automatically after batch inference)
# ---------------------------------------------------------------------------

def _export_tif_figures(pred_labels: np.ndarray,
                        confidences: np.ndarray,
                        wavelength_activation: np.ndarray) -> None:
    """
    Generate and save two publication-quality TIFF figures:

      fig_biomarker_activation_atlas.tif
          Wavelength-resolved MGAF structural activation curves per class,
          corresponding to manuscript Fig. 10(b).

      fig_confidence_distribution.tif
          Probability density histogram of AI prediction confidence per class,
          corresponding to manuscript Fig. 10(d).

    Formatting: Times New Roman, 300 DPI, sentence case labels.
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    x_ticks = np.arange(10)

    # --- Biomarker activation atlas ---
    fig1, ax1 = plt.subplots(figsize=(7, 5), dpi=300)
    for i, name in enumerate(CLASS_NAMES):
        mask = (pred_labels == i)
        if mask.sum() == 0:
            continue
        curve = np.mean(wavelength_activation[mask], axis=0)
        ax1.plot(x_ticks, curve, label=name, color=CLASS_COLORS[i],
                 linewidth=2.5, marker="o")
        ax1.fill_between(x_ticks, curve, alpha=0.08, color=CLASS_COLORS[i])
    ax1.set_xlabel("Wavelength (nm)", fontsize=12)
    ax1.set_ylabel("Mean MGAF structural activation intensity", fontsize=12)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(WAVELENGTHS, rotation=30)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=False)
    fig1.tight_layout()
    fig1.savefig("fig_biomarker_activation_atlas.tif",
                 dpi=300, format="tiff", bbox_inches="tight")
    plt.close(fig1)
    print("[EXPORT] fig_biomarker_activation_atlas.tif saved.")

    # --- Confidence density distribution ---
    fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=300)
    bins = np.linspace(0.5, 1.0, 21)
    for i, name in enumerate(CLASS_NAMES):
        mask = (pred_labels == i)
        if mask.sum() == 0:
            continue
        ax2.hist(confidences[mask], bins=bins, alpha=0.6,
                 label=name, color=CLASS_COLORS[i],
                 edgecolor=CLASS_COLORS[i], histtype="stepfilled", density=True)
    ax2.set_xlabel("AI confidence level", fontsize=12)
    ax2.set_ylabel("Probability density", fontsize=12)
    ax2.set_xlim(0.5, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=False)
    fig2.tight_layout()
    fig2.savefig("fig_confidence_distribution.tif",
                 dpi=300, format="tiff", bbox_inches="tight")
    plt.close(fig2)
    print("[EXPORT] fig_confidence_distribution.tif saved.")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/analyze-batch",
          summary="Batch hyperspectral inference",
          description=(
              "Upload PATCH and MGAF HDF5 feature files.  "
              "Returns macro population statistics and advanced analytics "
              "for dashboard visualisation.  Also saves SCI-grade TIFF figures."
          ))
async def analyze_batch(
    patch_file: UploadFile = File(..., description="PATCH feature library (.h5)"),
    mgaf_file:  UploadFile = File(..., description="MGAF structural matrix (.h5)"),
):
    """
    Main batch inference endpoint.

    Accepts dual-modal HDF5 feature files, executes GPU-batched HCTF inference,
    caches results in RESULT_DB, exports TIFF figures, and returns:
      - macro_stats           : predicted class counts and mean confidence
      - advanced_analytics    : per-class biomarker curves and confidence histograms
      - total_samples         : N (enables slider range in frontend)
    """
    patch_path = mgaf_path = ""
    try:
        # Write uploads to temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tp:
            tp.write(await patch_file.read())
            patch_path = tp.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tm:
            tm.write(await mgaf_file.read())
            mgaf_path = tm.name

        # Load feature arrays
        with h5py.File(patch_path, "r") as fp:
            raw_p = fp[list(fp.keys())[0]][()]
            x_patch = (np.squeeze(raw_p, axis=0)
                       if raw_p.ndim == 4 and raw_p.shape[0] == 1 else raw_p)
        with h5py.File(mgaf_path, "r") as fm:
            raw_m = fm[list(fm.keys())[0]][()]
            x_mgaf = (np.squeeze(raw_m, axis=0)
                      if raw_m.ndim == 5 and raw_m.shape[0] == 1 else raw_m)
        x_mgaf = np.nan_to_num(x_mgaf, nan=0.0)

        N = x_patch.shape[0]
        print(f"[INFERENCE] Batch inference on {N} samples (batch_size=128) ...")

        # GPU-batched forward pass — single call for all N samples
        outputs = _model.predict([x_patch, x_mgaf], batch_size=128, verbose=0)

        # Cache in memory database
        RESULT_DB["total_samples"] = N
        RESULT_DB["probs"]  = outputs["cls_output"]          # (N, 3)
        RESULT_DB["scores"] = outputs["mgaf_token_scores"]   # (N, 60)

        # Derive wavelength-resolved biomarker activation (MGAF diagonal mean)
        # x_mgaf: (N, 60, 10, 10) → diagonals: (N, 60, 10) → mean over patches: (N, 10)
        mgaf_diags = np.diagonal(x_mgaf, axis1=2, axis2=3)   # (N, 60, 10)
        wavelength_activation = np.mean(mgaf_diags, axis=1)  # (N, 10)

        pred_labels = np.argmax(RESULT_DB["probs"], axis=1)
        confidences = np.max(RESULT_DB["probs"], axis=1)
        counts = [int((pred_labels == i).sum()) for i in range(3)]

        # Automatically save SCI-grade publication figures
        _export_tif_figures(pred_labels, confidences, wavelength_activation)

        # --- Advanced analytics for frontend ECharts rendering ---

        # Per-class mean biomarker activation curves
        curves = {}
        for i in range(3):
            mask = (pred_labels == i)
            curves[i] = (np.mean(wavelength_activation[mask], axis=0).tolist()
                         if mask.sum() > 0 else [0.0] * 10)

        # Per-class confidence histograms (10 equal bins over [0.5, 1.0])
        conf_bins = np.linspace(0.5, 1.0, 11)
        distributions = {}
        for i in range(3):
            mask = (pred_labels == i)
            if mask.sum() > 0:
                counts_hist, _ = np.histogram(confidences[mask], bins=conf_bins)
                distributions[i] = (counts_hist / mask.sum()).tolist()
            else:
                distributions[i] = [0.0] * 10

        return {
            "status": "success",
            "total_samples": N,
            "macro_stats": {
                "counts": counts,
                "avg_confidence": float(confidences.mean()),
            },
            "advanced_analytics": {
                "wavelengths": WAVELENGTHS,
                "curves": curves,
                "confidence_bins": [
                    "50-55%", "55-60%", "60-65%", "65-70%", "70-75%",
                    "75-80%", "80-85%", "85-90%", "90-95%", "95-100%",
                ],
                "distributions": distributions,
            },
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

    finally:
        for path in (patch_path, mgaf_path):
            if path and os.path.exists(path):
                os.remove(path)


@app.get("/api/v1/sample/{sample_id}",
         summary="Per-kernel pathological traceability",
         description=(
             "Return the classification probabilities and 60-element MGAF token "
             "score vector for a specific sample ID.  Results are served from "
             "the in-memory cache populated by /api/v1/analyze-batch, so no "
             "additional model inference is triggered."
         ))
async def get_sample(sample_id: int):
    """
    Per-kernel micro-level query endpoint.

    The MGAF token score vector (length 60) is consumed by the frontend to
    render the spatial-spectral biomarker activation heatmap.  Reshaped to
    (10, 6), it maps to the peanut kernel's 10 × 6 patch grid.
    """
    if RESULT_DB["probs"] is None:
        return {"status": "error", "message": "No data in cache. Call /api/v1/analyze-batch first."}
    if not (0 <= sample_id < RESULT_DB["total_samples"]):
        return {"status": "error", "message": f"sample_id {sample_id} out of range "
                                               f"[0, {RESULT_DB['total_samples'] - 1}]."}
    return {
        "status": "success",
        "sample_id": sample_id,
        "prob":  RESULT_DB["probs"][sample_id].tolist(),    # [P_control, P_AC, P_NC]
        "score": RESULT_DB["scores"][sample_id].tolist(),   # 60 MGAF token scores
    }


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "SpectraX-Inspect is running. Visit /docs for the API reference."}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="SpectraX-Inspect API server")
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--port",    type=int, default=8000)
    parser.add_argument("--weights", default="hctf_best_deployment.weights.h5")
    args = parser.parse_args()

    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)