"""
Project high-dimensional model activations to the Eisenstein constraint manifold.

The constraint manifold is a hexagonal lattice in 2D integer space.
Every point is an Eisenstein integer z = a + bω where ω = e^(2πi/3).
Norm N(a,b) = a² - ab + b² defines distance from origin.

This module provides:
1. Affine transforms from model activation space → constraint manifold
2. Inverse transforms from constraint manifold → model activation space
3. Manifold-aware similarity (hexagonal distance, not Euclidean)
4. Batch projection for GPU-accelerated fleet communication
"""
import numpy as np


class ManifoldProjector:
    """
    Project model activations to the Eisenstein constraint manifold.

    The projection is a learned affine transform: z = Wx + b
    where x is the activation vector and z is a 2D manifold point.

    The transform maps the high-dimensional activation space to the
    2D Eisenstein lattice where:
    - Position encodes constraint state
    - Distance encodes semantic similarity
    - Norm encodes constraint satisfaction margin
    """

    def __init__(self, input_dim, manifold_radius=256):
        self.input_dim = input_dim
        self.manifold_radius = manifold_radius
        # Weight matrix: input_dim × 2 (project to 2D Eisenstein plane)
        self.W = np.random.randn(input_dim, 2) * 0.01
        self.b = np.zeros(2)
        self.fitted = False

    def fit(self, activations, constraint_states):
        """
        Fit the affine transform using least squares.

        activations: (N, input_dim) model activation vectors
        constraint_states: (N, 2) known Eisenstein coordinates for these activations
        """
        # Least squares: z = Wx + b
        # Pad activations with 1s for bias
        X = np.hstack([activations, np.ones((len(activations), 1))])
        Z = constraint_states
        # Solve: [W | b] = (X^T X)^-1 X^T Z
        W_padded = np.linalg.lstsq(X, Z, rcond=None)[0]
        self.W = W_padded[:-1]
        self.b = W_padded[-1]
        self.fitted = True

    def project(self, activations):
        """Project activations to manifold coordinates."""
        z = activations @ self.W + self.b
        # Snap to Eisenstein integers (nearest lattice point)
        a = np.round(z[:, 0]).astype(int)
        b = np.round(z[:, 1]).astype(int)
        # Clamp to manifold radius
        norm = a * a - a * b + b * b
        mask = norm > self.manifold_radius ** 2
        if mask.any():
            scale = self.manifold_radius / np.sqrt(norm[mask])
            a[mask] = (a[mask] * scale).astype(int)
            b[mask] = (b[mask] * scale).astype(int)
        return np.stack([a, b], axis=1)

    def inverse_project(self, manifold_points):
        """Project from manifold back to activation space."""
        # Pseudo-inverse: x = W^+ (z - b)
        W_pinv = np.linalg.pinv(self.W)  # (2, input_dim)
        z = manifold_points.astype(float)
        return (z - self.b) @ W_pinv

    def manifold_distance(self, z1, z2):
        """
        Eisenstein manifold distance (hexagonal metric).

        Unlike Euclidean distance, this respects the hexagonal symmetry
        of the constraint lattice.
        """
        da = z2[:, 0] - z1[:, 0]
        db = z2[:, 1] - z1[:, 1]
        # Eisenstein norm of difference
        return np.sqrt(da * da - da * db + db * db)

    def constraint_margin(self, z, radius_sq):
        """Check if manifold point is within constraint disk."""
        a, b = z[:, 0], z[:, 1]
        norm = a * a - a * b + b * b
        return radius_sq - norm  # positive = satisfied
