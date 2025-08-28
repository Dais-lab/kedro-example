"""Modeling pipeline definition."""
from kedro.pipeline import Pipeline, node

from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    """Create the modeling pipeline.
    
    Returns:
        A Pipeline object containing modeling nodes
    """
    return Pipeline(
        [
            node(
                func=train_model,
                inputs=["preprocessed_train_data", "parameters"],
                outputs=["trained_model", "training_metrics"],
                name="05_train_cnn_model",
            ),

        ]
    )
