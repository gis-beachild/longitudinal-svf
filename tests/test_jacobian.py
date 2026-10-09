"""Numerical checks for Jacobians of known affine deformations."""

import pytest
import torch

from src.losses.jacobian import (
    Jacobianloss,
    compute_jacobian_determinant,
    compute_jacobian_determinant_3d,
    jacobian_determinant_3d,
)
from src.utils.grid_utils import displacement2grid


@pytest.mark.parametrize(
    "matrix",
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.2, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.3]],
        [[0.0, 0.2, 0.3], [-0.4, 0.0, 0.1], [0.2, 0.1, 0.0]],
        [[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ],
)
def test_affine_jacobian_matches_matrix_determinant(matrix):
    matrix = torch.tensor(matrix, dtype=torch.float64)
    coordinates = torch.stack(torch.meshgrid(
        torch.arange(5, dtype=torch.float64),
        torch.arange(6, dtype=torch.float64),
        torch.arange(7, dtype=torch.float64),
        indexing="ij",
    ))
    displacement = torch.einsum("ij,jdhw->idhw", matrix, coordinates)
    expected = torch.linalg.det(torch.eye(3, dtype=torch.float64) + matrix)

    direct = compute_jacobian_determinant_3d(displacement)
    batched = compute_jacobian_determinant_3d(displacement.unsqueeze(0))
    normalized_grid = displacement2grid(displacement.unsqueeze(0))
    grid_det = compute_jacobian_determinant(normalized_grid)

    assert direct.shape == (5, 6, 7)
    assert batched.shape == (1, 5, 6, 7)
    assert grid_det.shape == (1, 4, 5, 6)
    assert torch.allclose(direct, torch.full_like(direct, expected))
    assert torch.allclose(batched[0], direct)
    assert torch.allclose(grid_det, torch.full_like(grid_det, expected))
    assert torch.allclose(jacobian_determinant_3d(normalized_grid), grid_det)
    expected_penalty = torch.clamp(-expected, min=0) * displacement.shape[1:].numel()
    assert torch.allclose(Jacobianloss()(displacement.unsqueeze(0)), expected_penalty)


def test_physical_spacing_for_physical_displacement():
    spacing = (2.0, 3.0, 4.0)
    coordinates = torch.stack(torch.meshgrid(
        torch.arange(5, dtype=torch.float64) * spacing[0],
        torch.arange(6, dtype=torch.float64) * spacing[1],
        torch.arange(7, dtype=torch.float64) * spacing[2],
        indexing="ij",
    ))
    displacement = torch.zeros_like(coordinates)
    displacement[0] = 0.25 * coordinates[0]
    displacement[2] = -0.2 * coordinates[2]
    determinant = compute_jacobian_determinant_3d(displacement, spacing=spacing)
    assert torch.allclose(determinant, torch.full_like(determinant, 1.25 * 0.8))
