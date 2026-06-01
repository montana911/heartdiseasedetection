"""
heart disease detection using hybrid machine learning 
=======================================

Bachelor thesis implementation of a hybrid ensemble machine learning
pipeline for cardiovascular disease screening on BRFSS survey data.

Module layout
-------------
- config             central constants and hyperparameter values
- data_loader        loading and merging the two CDC survey cycles
- preprocessor       feature encoding, scaling, and partitioning
- imbalance_handler  SMOTE oversampling with leakage protection
- models             base classifier factory and the manual soft-voting ensemble
- evaluator          custom metric computation with imbalance-aware reporting
- visualizer         all results plots used in the thesis
- pipeline           end-to-end orchestrator that composes the modules above

----------
"""

from .config       import Config
from .data_loader  import DatasetLoader
from .preprocessor import FeaturePreprocessor
from .imbalance_handler import ImbalanceHandler
from .models       import ModelFactory, SoftVotingEnsemble
from .evaluator    import ModelEvaluator
from .visualizer   import ResultsVisualizer
from .pipeline     import HeartDiseasePipeline

__version__ = "1.0.0"
__author__  = "Bachelor Thesis Project"

__all__ = [
    "Config",
    "DatasetLoader",
    "FeaturePreprocessor",
    "ImbalanceHandler",
    "ModelFactory",
    "SoftVotingEnsemble",
    "ModelEvaluator",
    "ResultsVisualizer",
    "HeartDiseasePipeline",
]
