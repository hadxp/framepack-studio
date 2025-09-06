"""
Single Frame pipeline for FramePack Studio.

This pipeline produces a single generated frame by reusing the OriginalPipeline
preprocessing and forcing parameters to sample exactly one latent frame.
"""

from .original_pipeline import OriginalPipeline


class SingleFramePipeline(OriginalPipeline):
    """Pipeline for Single Frame generation type."""

    def prepare_parameters(self, job_params):
        """
        Prepare parameters for the Single Frame generation job.

        Forces latent_window_size to 1 so the worker samples a single frame,
        and sets total_second_length to the duration of one latent section
        (latent_window_size * 4 / 30 seconds) to ensure the pipeline/worker
        computes at least one latent section.
        """
        processed_params = job_params.copy()

        # Ensure correct model type label
        processed_params["model_type"] = "Single Frame"

        # Force a single latent window so sampling produces exactly one frame
        processed_params["latent_window_size"] = 1

        # One latent section duration (1 * 4 frames / 30 fps)
        # This ensures the worker computes at least one latent section.
        processed_params["total_second_length"] = 4.0 / 30.0

        return processed_params

    def validate_parameters(self, job_params):
        """
        Validate parameters for the Single Frame generation job.

        Similar to OriginalPipeline, but allows the very short total_second_length
        we set in prepare_parameters.
        """
        # Reuse the OriginalPipeline validation for most checks
        is_valid, error = super().validate_parameters(job_params)
        if not is_valid:
            return is_valid, error

        # Steps must be > 0 (Original validation already checks this), keep same behavior
        if job_params.get("steps", 0) <= 0:
            return False, "Steps must be greater than 0"

        return True, None

    # preprocess_inputs and handle_results are inherited from OriginalPipeline
