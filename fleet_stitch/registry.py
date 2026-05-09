"""Stitching matrix registry for cross-model communication via the constraint manifold."""


class StitchRegistry:
    """
    Registry of pre-computed stitching matrices between model pairs.

    Stores affine transforms keyed by (source_model, target_model, layer_depth).
    When two models need to communicate, look up the stitching matrix
    and project through it.
    """

    def __init__(self):
        self.matrices = {}

    def register(self, source_model, target_model, projector):
        """Register a stitching matrix for a model pair."""
        key = (source_model, target_model)
        self.matrices[key] = projector

    def get(self, source_model, target_model):
        """Get the stitching matrix for a model pair."""
        return self.matrices.get((source_model, target_model))

    def can_stitch(self, source_model, target_model):
        """Check if a stitching matrix exists for this pair."""
        return (source_model, target_model) in self.matrices

    def stitch(self, source_model, target_model, activations):
        """
        Project from source model's space to target model's space
        via the constraint manifold.
        """
        source_to_manifold = self.get(source_model, 'manifold')
        manifold_to_target = self.get('manifold', target_model)

        if source_to_manifold and manifold_to_target:
            manifold_points = source_to_manifold.project(activations)
            return manifold_to_target.inverse_project(manifold_points)
        return None
