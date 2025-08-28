"""Data processing pipeline definition."""
from kedro.pipeline import Pipeline, node

from .nodes import (
    load_raw_data,
    load_test_data,
    preprocess_data,
    preprocess_test_data
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data processing pipeline.
    
    Returns:
        A Pipeline object containing data processing nodes
    """
    return Pipeline(
        [
            node(
                func=load_raw_data,
                inputs="params:data_path",
                outputs="raw_train_data",
                name="01_load_raw_training_data",
            ),
            node(
                func=load_test_data,
                inputs="params:data_path", 
                outputs="raw_test_data",
                name="02_load_raw_test_data",
            ),
            node(
                func=preprocess_data,
                inputs=["raw_train_data", "parameters"],
                outputs="preprocessed_train_data",
                name="03_preprocess_training_data",
            ),
            node(
                func=preprocess_test_data,
                inputs=["raw_test_data", "parameters"],
                outputs="preprocessed_test_data",
                name="04_preprocess_test_data",
            ),
        ]
    )
