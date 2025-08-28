"""Inference pipeline definition."""
from kedro.pipeline import Pipeline, node

from .nodes import make_predictions, evaluate_predictions


def create_pipeline(**kwargs) -> Pipeline:
    """Create the inference pipeline.
    
    Returns:
        A Pipeline object containing inference nodes
    """
    return Pipeline(
        [
            node(
                func=make_predictions,
                inputs=["trained_model", "preprocessed_test_data", "parameters"],
                outputs="predictions",
                name="06_make_predictions",
            ),
            node(
                func=evaluate_predictions,
                inputs=["predictions", "preprocessed_test_data"],
                outputs="evaluation_report",
                name="07_evaluate_predictions",
            ),
        ]
    )
