# fleet-stitch — Constraint Manifold Projection for Model Communication

Project model activations to the Eisenstein constraint manifold.
Communicate between models without tokenization.

## Why
Latent space translation lets models share thoughts at the vector level.
But learned latent spaces drift, are model-specific, and are black boxes.

The Eisenstein constraint manifold is:
- Mathematically defined (not learned)
- Deterministic (same activations → same manifold point)
- Cross-model (any model can project to it)
- Discrete (integer arithmetic, no float drift)
- Interpretable (every point has algebraic meaning)

## How It Works
1. Fit: Learn affine transform from model activations → manifold (one-time)
2. Project: Any activation → Eisenstein integer (a, b) on the lattice
3. Communicate: Transmit (a, b) instead of the full activation vector
4. Inverse: Receiving model projects back to its own activation space

## The Stack
```
Model A activations (4096-dim)
    → affine_A (2-dim Eisenstein point)
    → transmit (2 integers)
    → affine_B_inv (4096-dim)  
    → Model B continues reasoning
```

## Composable With
- casting-call-gpu (Oracle1): voice signatures → manifold coordinates
- plato-vector-persistence: store manifold points alongside embeddings
- fleet-constraint-kernel: GPU-accelerated batch projection
- temporal-flux: T_SNAP opcode snaps manifold projections
- physics-clock: temporal dimensions on manifold points
- insight-cfp-bridge: share manifold discoveries as FLUX tiles

## Install
```bash
pip install fleet-stitch
```

## Quick Start
```python
from fleet_stitch import ManifoldProjector, StitchRegistry

# Fit a projector for your model
proj = ManifoldProjector(input_dim=4096)
proj.fit(training_activations, known_constraint_states)

# Project new activations to manifold
manifold_points = proj.project(new_activations)  # (N, 2) integer array

# Register for cross-model stitching
registry = StitchRegistry()
registry.register('my_model', 'manifold', proj)

# Stitch from one model to another
target_activations = registry.stitch('model_a', 'model_b', source_activations)
```

## License
Apache 2.0
