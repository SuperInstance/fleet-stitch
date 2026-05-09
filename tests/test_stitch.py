"""Tests for fleet-stitch constraint manifold projection."""
import numpy as np
import pytest
from fleet_stitch import ManifoldProjector, StitchRegistry


class TestManifoldProjector:
    def test_weight_dimensions(self):
        """Projector creates correct weight dimensions."""
        p = ManifoldProjector(input_dim=128)
        assert p.W.shape == (128, 2)
        assert p.b.shape == (2,)

    def test_fit_and_project_integers(self):
        """Fit + project produces integer Eisenstein coordinates."""
        np.random.seed(42)
        p = ManifoldProjector(input_dim=16)
        acts = np.random.randn(50, 16)
        states = np.random.randint(-10, 10, size=(50, 2)).astype(float)
        p.fit(acts, states)
        assert p.fitted
        result = p.project(acts)
        assert result.dtype == int
        assert result.shape == (50, 2)

    def test_hexagonal_symmetry(self):
        """Manifold distance respects hexagonal symmetry."""
        p = ManifoldProjector(input_dim=4)
        z1 = np.array([[0, 0]])
        # Verify Eisenstein norm: N(a,b) = a² - ab + b²
        # (1,0) → N=1, d=1; (0,1) → N=1, d=1; (1,1) → N=1, d=1
        for pt in [(1, 0), (0, 1), (1, 1)]:
            z2 = np.array([pt])
            d = p.manifold_distance(z1, z2)
            np.testing.assert_almost_equal(d[0], 1.0, decimal=10)

    def test_constraint_margin(self):
        """Points inside disk = positive margin, outside = negative."""
        p = ManifoldProjector(input_dim=4)
        inside = np.array([[1, 0]])
        outside = np.array([[100, 0]])
        r2 = 10.0
        assert p.constraint_margin(inside, r2) > 0
        assert p.constraint_margin(outside, r2) < 0

    def test_round_trip(self):
        """Inverse projection approximately preserves structure."""
        np.random.seed(99)
        dim = 8
        p = ManifoldProjector(input_dim=dim)
        acts = np.random.randn(20, dim)
        states = np.random.randn(20, 2) * 3
        p.fit(acts, states)
        projected = p.project(acts)
        recovered = p.inverse_project(projected.astype(float))
        # Low-rank projection loses info, but structure should correlate
        corr = np.corrcoef(acts.flatten(), recovered.flatten())[0, 1]
        assert abs(corr) > 0.1  # some structure preserved

    def test_batch_projection(self):
        """N activations → N manifold points."""
        np.random.seed(7)
        p = ManifoldProjector(input_dim=32)
        acts = np.random.randn(100, 32)
        states = np.random.randn(100, 2)
        p.fit(acts, states)
        result = p.project(acts)
        assert result.shape == (100, 2)

    def test_manifold_radius_clamp(self):
        """Points outside manifold radius get clamped."""
        np.random.seed(11)
        p = ManifoldProjector(input_dim=4, manifold_radius=5)
        acts = np.random.randn(20, 4)
        states = np.random.randn(20, 2) * 2
        p.fit(acts, states)
        pts = p.project(acts)
        norms = pts[:, 0] ** 2 - pts[:, 0] * pts[:, 1] + pts[:, 1] ** 2
        # All norms should be ≤ manifold_radius^2
        assert np.all(norms <= p.manifold_radius ** 2)


class TestStitchRegistry:
    def test_register_and_get(self):
        """Register and retrieve a stitching matrix."""
        reg = StitchRegistry()
        p = ManifoldProjector(input_dim=8)
        reg.register('model_a', 'model_b', p)
        assert reg.get('model_a', 'model_b') is p

    def test_can_stitch(self):
        """can_stitch returns True only for registered pairs."""
        reg = StitchRegistry()
        p = ManifoldProjector(input_dim=8)
        reg.register('model_a', 'model_b', p)
        assert reg.can_stitch('model_a', 'model_b')
        assert not reg.can_stitch('model_b', 'model_a')

    def test_stitch_pipeline(self):
        """Full stitch: source → manifold → target."""
        np.random.seed(33)
        dim = 8
        reg = StitchRegistry()
        src_to_m = ManifoldProjector(input_dim=dim)
        m_to_tgt = ManifoldProjector(input_dim=dim)

        acts = np.random.randn(30, dim)
        states = np.random.randn(30, 2) * 3
        src_to_m.fit(acts, states)
        m_to_tgt.fit(acts, states)

        reg.register('src', 'manifold', src_to_m)
        reg.register('manifold', 'tgt', m_to_tgt)

        result = reg.stitch('src', 'tgt', acts)
        assert result is not None
        assert result.shape == (30, dim)

    def test_stitch_missing_returns_none(self):
        """Stitch with missing matrices returns None."""
        reg = StitchRegistry()
        result = reg.stitch('a', 'b', np.random.randn(5, 8))
        assert result is None
