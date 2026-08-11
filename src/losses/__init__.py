"""Similarity and regularization loss functions used to train the registration networks.

Author: Fl0rian
"""
from .gradient import Grad3d
from .magnitude import MagnitudeLoss
from .jacobian import Jacobianloss
from .inverse_consistency import InverseConsistency, IconInverseConsistency, GradIconInverseConsistency