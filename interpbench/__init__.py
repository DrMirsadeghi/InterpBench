"""
InterpBench
===========

A benchmarking framework for deep neural networks that unifies

    • Generalization
    • Robustness
    • Interpretability
    • Representation Geometry

into a single, standardized evaluation pipeline.

Public API
----------

Evaluate a model

    >>> import interpbench as ib
    >>> report = ib.evaluate(model, testloader)

Train while benchmarking

    >>> trainer = ib.StandardTrainer(...)
    >>> trainer.fit(...)

Visualize benchmark results

    >>> ib.plot(report)

The canonical object of InterpBench is the Report.
Every public function either produces, consumes, or transforms a Report.
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

from .evaluate import evaluate

from .trainers import (
    StandardTrainer,
    PGDTrainer,
)

from .plot import plot

from .report import (
    Report,
    Snapshot,
    LayerReport,
)

# ---------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------

__all__ = [

    # evaluation
    "evaluate",

    # visualization
    "plot",

    # trainers
    "StandardTrainer",
    "PGDTrainer",

    # report hierarchy
    "Report",
    "Snapshot",
    "LayerReport",
]