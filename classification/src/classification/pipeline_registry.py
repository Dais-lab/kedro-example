"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

# Import pipeline creation functions
from classification.pipelines import data_processing
from classification.pipelines import modeling
from classification.pipelines import inference


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Create individual pipelines
    data_processing_pipeline = data_processing.create_pipeline()
    modeling_pipeline = modeling.create_pipeline()
    inference_pipeline = inference.create_pipeline()
    
    # Register pipelines
    pipelines = {
        "data_processing": data_processing_pipeline,
        "modeling": modeling_pipeline,
        "inference": inference_pipeline,
        "__default__": data_processing_pipeline + modeling_pipeline + inference_pipeline,
    }
    
    return pipelines
