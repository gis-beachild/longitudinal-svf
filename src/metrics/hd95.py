"""Surface Hausdorff distance computed from binary segmentations."""

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree


def hd95_mm(pred_mask, reference_mask, pred_affine, reference_affine):
    """Return max(P95(pred→reference), P95(reference→pred)) in mm.

    Masks are 3D binary arrays. Each 4x4 affine maps its voxel indices to
    a common physical coordinate system in millimetres (e.g. TorchIO affines).
    Surface samples are centres of foreground boundary voxels, using
    6-connectivity and treating voxels outside the image as background.
    Both empty masks return 0; exactly one empty mask returns infinity.
    Separate affines allow different spacings, orientations and origins.
    """
    surfaces = []
    for mask, affine in ((pred_mask, pred_affine), (reference_mask, reference_affine)):
        mask = np.asarray(mask, dtype=bool)
        affine = np.asarray(affine, dtype=float)
        if mask.ndim != 3:
            raise ValueError("HD95 requires 3D binary masks")
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("HD95 requires finite 4x4 affines in millimetres")
        boundary = mask & ~binary_erosion(
            mask, structure=generate_binary_structure(3, 1), border_value=0
        )
        points = np.argwhere(boundary)
        surfaces.append(points @ affine[:3, :3].T + affine[:3, 3])

    pred_surface, reference_surface = surfaces
    if not len(pred_surface) or not len(reference_surface):
        return 0.0 if len(pred_surface) == len(reference_surface) else float("inf")
    pred_to_ref = cKDTree(reference_surface).query(pred_surface)[0]
    ref_to_pred = cKDTree(pred_surface).query(reference_surface)[0]
    return float(max(np.percentile(pred_to_ref, 95), np.percentile(ref_to_pred, 95)))
