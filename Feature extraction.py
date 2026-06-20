"""
feature_extraction.py
=====================
Production-grade hyperspectral feature extraction pipeline.

Implements the three-stage preprocessing described in manuscript Sections 2.2-2.4:

  Stage 1 — ROI segmentation (Section 2.2, Fig. 1):
    Otsu thresholding + connected-component analysis on the reference-band
    grayscale image to isolate individual peanut kernel regions from the
    hyperspectral acquisition tray, then normalize each ROI to a fixed
    spatial extent (50 × 30 × 206).

  Stage 2 — PCA-guided PATCH features (Section 2.3, Eq. 1):
    Partition each ROI into non-overlapping 5 × 5 spatial patches and compute
    the mean spectrum per patch (60 patches per kernel).  Fit PCA globally on
    the full training corpus to select the top-10 most informative spectral
    bands; apply the fixed band indices at inference time without re-fitting
    to guarantee training/deployment consistency.

  Stage 3 — MGAF structural features (Section 2.4, Eq. 2-4):
    Transform each 10-band patch spectrum into a 10 × 10 Multi-dimensional
    Gramian Angular Field matrix, encoding pairwise nonlinear spectral
    correlations weighted by Gaussian band-proximity kernels.

Design principles:
  * PCA band selection is fit once offline and serialised as a plain dict
    (band_indices + wavelengths_nm).  The deployment path only needs these
    indices; it never re-fits PCA, ensuring inference-time reproducibility.
  * All computation is pure NumPy / SciPy / scikit-image — no TensorFlow
    dependency — enabling lightweight edge-device preprocessing.
  * NaN / Inf values in MGAF matrices are replaced with 0 before persistence
    and model ingestion.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.transform import resize


# ---------------------------------------------------------------------------
# Global constants — mirror manuscript Section 2.1 / 2.2
# ---------------------------------------------------------------------------

N_BANDS_RAW          = 206          # raw spectral bands (1000–2216 nm)
WAVELENGTH_START_NM  = 1000.0       # first band centre wavelength (nm)
WAVELENGTH_END_NM    = 2216.0       # last band centre wavelength (nm)

TARGET_H, TARGET_W   = 50, 30       # normalised kernel ROI size (px)
PATCH_SIZE           = 5            # spatial patch side length (px)
N_PATCHES            = (TARGET_H // PATCH_SIZE) * (TARGET_W // PATCH_SIZE)  # 60
PATCH_GRID_SHAPE     = (TARGET_H // PATCH_SIZE, TARGET_W // PATCH_SIZE)      # (10, 6)

N_PCA_BANDS          = 10           # selected bands after PCA
MGAF_SIGMA_DEFAULT   = 1.5         # Gaussian proximity kernel sigma (Eq. 3)

# Biomarker window identified in manuscript Section 3.5.3
BIOMARKER_WINDOW_NM  = (2159.0, 2216.0)


def band_index_to_wavelength(band_idx) -> np.ndarray:
    """Convert zero-based band indices to physical wavelengths (nm)."""
    idx = np.asarray(band_idx, dtype=float)
    return WAVELENGTH_START_NM + idx * (
        (WAVELENGTH_END_NM - WAVELENGTH_START_NM) / (N_BANDS_RAW - 1)
    )


# ---------------------------------------------------------------------------
# Stage 1: ROI segmentation (manuscript Section 2.2, Fig. 1c-f)
# ---------------------------------------------------------------------------

def segment_kernels_from_cube(
    hyperspectral_cube: np.ndarray,
    reference_band_idx: Optional[int] = None,
    reference_wavelength_nm: float = 1133.1,
    min_area_px: int = 60,
) -> List[np.ndarray]:
    """
    Segment individual peanut kernel ROIs from a raw hyperspectral tray image.

    Implements the pipeline illustrated in manuscript Fig. 1(c)-(f):
      (c) Select reference-band reflectance image
      (d) Apply Otsu thresholding to generate binary foreground mask
      (e) Apply mask to full-spectrum cube
      (f) Extract per-kernel bounding boxes via connected-component centroids;
          normalise each kernel ROI to (TARGET_H, TARGET_W, N_BANDS_RAW).

    Parameters
    ----------
    hyperspectral_cube      : ndarray of shape (H, W, N_BANDS_RAW)
                              White/dark reference-corrected raw cube.
    reference_band_idx      : int, optional
                              Zero-based band index for Otsu segmentation.
                              Computed from reference_wavelength_nm if None.
    reference_wavelength_nm : float
                              Physical wavelength (nm) of the reference band
                              used for segmentation (default: 1133.1 nm).
    min_area_px             : int
                              Minimum connected-component area (pixels) to
                              accept as a genuine kernel region.

    Returns
    -------
    list of ndarray, each of shape (TARGET_H, TARGET_W, N_BANDS_RAW)
        One element per segmented kernel ROI.
    """
    H, W, B = hyperspectral_cube.shape

    if reference_band_idx is None:
        reference_band_idx = int(round(
            (reference_wavelength_nm - WAVELENGTH_START_NM)
            / (WAVELENGTH_END_NM - WAVELENGTH_START_NM) * (B - 1)
        ))
        reference_band_idx = int(np.clip(reference_band_idx, 0, B - 1))

    # Binary foreground mask via Otsu thresholding
    ref_image = hyperspectral_cube[:, :, reference_band_idx]
    thresh = threshold_otsu(ref_image)
    binary_mask = (ref_image > thresh).astype(np.uint8)

    labeled_mask = label(binary_mask, connectivity=2)
    rois: List[np.ndarray] = []

    for region in regionprops(labeled_mask):
        if region.area < min_area_px:
            continue
        r0, c0, r1, c1 = region.bbox
        sub_cube = hyperspectral_cube[r0:r1, c0:c1, :].copy()

        # Zero non-kernel pixels within bounding box
        kernel_mask = (labeled_mask[r0:r1, c0:c1] == region.label)
        sub_cube *= kernel_mask[:, :, None]

        # Spatial normalisation with bilinear interpolation
        resized = resize(
            sub_cube, (TARGET_H, TARGET_W, B),
            order=1, preserve_range=True, anti_aliasing=False,
        ).astype(np.float32)
        rois.append(resized)

    return rois


# ---------------------------------------------------------------------------
# Stage 2a: Spatial patch mean spectra (manuscript Section 2.3, Eq. 1)
# ---------------------------------------------------------------------------

def extract_patch_mean_spectra(
    roi_cube: np.ndarray,
    patch_size: int = PATCH_SIZE,
) -> np.ndarray:
    """
    Partition a kernel ROI into non-overlapping spatial patches and compute
    the mean spectrum for each patch (manuscript Eq. 1):

        x_avg = (1/n) * Σ_{i=1}^{n} x_i,   n = patch_size²

    Parameters
    ----------
    roi_cube   : ndarray of shape (TARGET_H, TARGET_W, B)
    patch_size : spatial side length of each patch (default: 5)

    Returns
    -------
    ndarray of shape (N_PATCHES, B)
        60 mean spectral vectors per kernel when patch_size=5.
    """
    H, W, B = roi_cube.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(
            f"ROI dimensions ({H}, {W}) must be divisible by patch_size={patch_size}. "
            f"Ensure the cube is normalised to ({TARGET_H}, {TARGET_W})."
        )
    n_rows, n_cols = H // patch_size, W // patch_size
    # Reshape to expose patch spatial axes, then average them
    patches = roi_cube.reshape(n_rows, patch_size, n_cols, patch_size, B)
    patch_means = patches.mean(axis=(1, 3))          # (n_rows, n_cols, B)
    return patch_means.reshape(n_rows * n_cols, B)   # (N_PATCHES, B)


# ---------------------------------------------------------------------------
# Stage 2b: PCA-guided band selection (manuscript Section 2.3)
# ---------------------------------------------------------------------------

@dataclass
class PCABandSelector:
    """
    Offline PCA-guided spectral band selector.

    Training phase:
      Call fit() on the concatenated patch spectra from all training kernels
      (shape: N_total_patches × N_BANDS_RAW).  Internally runs sklearn PCA,
      sums absolute loadings across the first n_components principal components,
      and retains the top n_bands wavelengths.

    Inference phase:
      Restore via from_dict() and call transform() — no PCA re-fitting required.
      This guarantees that the same 10 band indices are applied to every sample
      throughout training, validation, and deployment (see manuscript Section 2.3).

    Serialisation:
      Use to_dict() / from_dict() to save and restore band indices alongside
      model weights.

    Parameters
    ----------
    n_components : number of PCA principal components to consider
    n_bands      : number of top-importance bands to select
    """

    n_components: int = 10
    n_bands: int = N_PCA_BANDS
    band_indices_: Optional[np.ndarray] = field(default=None, repr=False)
    wavelengths_nm_: Optional[np.ndarray] = field(default=None, repr=False)
    explained_variance_ratio_: Optional[np.ndarray] = field(default=None, repr=False)

    def fit(self, all_patch_spectra: np.ndarray) -> "PCABandSelector":
        """
        Fit band selector on full training corpus.

        Parameters
        ----------
        all_patch_spectra : ndarray of shape (N_total_patches, N_BANDS_RAW)
            Concatenated mean patch spectra across all training kernels.
        """
        from sklearn.decomposition import PCA

        pca = PCA(n_components=self.n_components)
        pca.fit(all_patch_spectra)

        # Sum absolute loadings across all retained PCs → global band importance
        importance = np.abs(pca.components_).sum(axis=0)   # (N_BANDS_RAW,)
        top_idx = np.argsort(importance)[::-1][: self.n_bands]
        top_idx = np.sort(top_idx)   # ascending wavelength order for readability

        self.band_indices_ = top_idx
        self.wavelengths_nm_ = band_index_to_wavelength(top_idx)
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        return self

    def transform(self, patch_spectra: np.ndarray) -> np.ndarray:
        """
        Apply fixed band selection (last axis indexed).

        Parameters
        ----------
        patch_spectra : ndarray (..., N_BANDS_RAW)

        Returns
        -------
        ndarray (..., n_bands)
        """
        if self.band_indices_ is None:
            raise RuntimeError("Call fit() or from_dict() before transform().")
        return patch_spectra[..., self.band_indices_]

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict for persistence alongside weights."""
        return {
            "band_indices": self.band_indices_.tolist(),
            "wavelengths_nm": self.wavelengths_nm_.tolist(),
            "n_components": self.n_components,
            "n_bands": self.n_bands,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PCABandSelector":
        """Restore from a serialised dict (no re-fitting required)."""
        obj = cls(n_components=d["n_components"], n_bands=d["n_bands"])
        obj.band_indices_ = np.array(d["band_indices"], dtype=int)
        obj.wavelengths_nm_ = np.array(d["wavelengths_nm"], dtype=float)
        return obj


# ---------------------------------------------------------------------------
# Stage 3: MGAF spectral structural features (manuscript Section 2.4, Eq. 2-4)
# ---------------------------------------------------------------------------

def compute_mgaf_matrix(spectrum_10band: np.ndarray,
                        sigma: float = MGAF_SIGMA_DEFAULT) -> np.ndarray:
    """
    Compute the MGAF matrix for a single 10-band patch spectrum.

    Implements manuscript Equations 2-4:

      x_hat   = (x_band − μ) / (σ_std + 1e-6)                    (Eq. 2)
      W_ij    = exp(−(i−j)² / (2σ²))                               (Eq. 3)
      MGAF_ij = W_ij · x̂_i · x̂_j                                (Eq. 4)

    The Gaussian weight W_ij encodes spectral proximity: adjacent bands are
    strongly coupled, distant bands are weakly coupled.  The outer product
    x̂_i · x̂_j captures pairwise normalised reflectance interactions.

    Parameters
    ----------
    spectrum_10band : ndarray of shape (10,)
    sigma           : Gaussian proximity kernel scale parameter

    Returns
    -------
    ndarray of shape (10, 10), dtype float32
    """
    c = spectrum_10band.shape[-1]   # should be 10
    mu = spectrum_10band.mean()
    std = spectrum_10band.std()
    x_hat = (spectrum_10band - mu) / (std + 1e-6)   # normalised (Eq. 2)

    idx = np.arange(c, dtype=float)
    diff_sq = (idx[:, None] - idx[None, :]) ** 2
    W = np.exp(-diff_sq / (2.0 * sigma ** 2))       # Gaussian kernel (Eq. 3)

    mgaf = W * np.outer(x_hat, x_hat)               # pairwise product (Eq. 4)
    return mgaf.astype(np.float32)


def build_mgaf_features(patch_spectra_10band: np.ndarray,
                        sigma: float = MGAF_SIGMA_DEFAULT) -> np.ndarray:
    """
    Compute MGAF matrices for all patches of one kernel.

    Parameters
    ----------
    patch_spectra_10band : ndarray of shape (N_PATCHES, 10)

    Returns
    -------
    ndarray of shape (N_PATCHES, 10, 10), NaN/Inf replaced with 0.
    """
    n = patch_spectra_10band.shape[0]
    out = np.empty((n, 10, 10), dtype=np.float32)
    for i in range(n):
        out[i] = compute_mgaf_matrix(patch_spectra_10band[i], sigma=sigma)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Convenience: end-to-end single-kernel feature extractor
# ---------------------------------------------------------------------------

@dataclass
class SampleFeatures:
    """Container for the dual-modal features of a single peanut kernel."""
    x_patch: np.ndarray          # (60, 10)  — PATCH features
    x_mgaf: np.ndarray           # (60, 10, 10) — MGAF features
    wavelengths_nm: np.ndarray   # (10,) — selected band physical wavelengths


def build_sample_features(
    roi_cube: np.ndarray,
    band_selector: PCABandSelector,
    sigma: float = MGAF_SIGMA_DEFAULT,
) -> SampleFeatures:
    """
    End-to-end single-kernel feature extraction pipeline.

    Stages: ROI (50,30,206)
      → patch mean spectra (60, 206)   [Eq. 1]
      → PCA band selection  (60, 10)   [fixed indices]
      → MGAF matrices       (60,10,10) [Eq. 2-4]

    Parameters
    ----------
    roi_cube      : normalised kernel cube (TARGET_H, TARGET_W, N_BANDS_RAW)
    band_selector : fitted or restored PCABandSelector
    sigma         : MGAF Gaussian kernel scale

    Returns
    -------
    SampleFeatures with x_patch, x_mgaf, and wavelengths_nm populated.
    """
    full_spectra = extract_patch_mean_spectra(roi_cube)          # (60, 206)
    x_patch = band_selector.transform(full_spectra)              # (60, 10)
    x_mgaf = build_mgaf_features(x_patch, sigma=sigma)          # (60, 10, 10)
    return SampleFeatures(
        x_patch=x_patch.astype(np.float32),
        x_mgaf=x_mgaf,
        wavelengths_nm=band_selector.wavelengths_nm_,
    )


def reshape_patch_scores_to_grid(scores_60: np.ndarray) -> np.ndarray:
    """
    Reshape a per-patch score vector (length 60) to the kernel spatial grid
    (10 rows × 6 cols) for heatmap overlay visualisation.

    Parameters
    ----------
    scores_60 : ndarray of shape (..., 60)

    Returns
    -------
    ndarray of shape (..., 10, 6)
    """
    return scores_60.reshape(*scores_60.shape[:-1], *PATCH_GRID_SHAPE)


# ---------------------------------------------------------------------------
# Self-test (runs without any real data files)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(seed=0)
    print("[self-test] Generating synthetic hyperspectral cube ...")
    fake_cube = rng.normal(0.15, 0.03, (120, 200, N_BANDS_RAW)).astype(np.float32)
    # Embed three brighter kernel-like regions for Otsu to find
    for (r0, c0) in [(10, 10), (60, 80), (20, 140)]:
        fake_cube[r0:r0 + 45, c0:c0 + 28, :] += 0.4

    rois = segment_kernels_from_cube(fake_cube)
    print(f"[self-test] Segmented {len(rois)} kernel ROI(s); "
          f"first shape: {rois[0].shape if rois else 'N/A'}")

    all_patches = np.concatenate(
        [extract_patch_mean_spectra(r) for r in rois], axis=0
    )
    selector = PCABandSelector().fit(all_patches)
    print(f"[self-test] Selected band indices : {selector.band_indices_}")
    print(f"[self-test] Corresponding wavelengths (nm): "
          f"{np.round(selector.wavelengths_nm_, 1)}")

    feats = build_sample_features(rois[0], selector)
    assert feats.x_patch.shape == (N_PATCHES, N_PCA_BANDS), "PATCH shape mismatch"
    assert feats.x_mgaf.shape == (N_PATCHES, N_PCA_BANDS, N_PCA_BANDS), \
        "MGAF shape mismatch"
    print(f"[self-test] x_patch: {feats.x_patch.shape}  "
          f"x_mgaf: {feats.x_mgaf.shape}")
    print("[self-test] PASS ✓  All assertions satisfied.")