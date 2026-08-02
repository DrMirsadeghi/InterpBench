"""
report.py

Canonical data structures for InterpBench.

Everything in InterpBench either

    • produces a Report
    • consumes a Report
    • transforms a Report

A Report is simply an ordered collection of Snapshots.

A Snapshot represents one benchmark evaluation. Every Snapshot carries
canonical experiment metadata

    model
    dataset
    trainer
    run
    epoch

which together identify the history to which the Snapshot belongs.

The legacy ``metadata`` dictionary is preserved for backward
compatibility. The canonical fields and ``metadata`` are automatically
kept synchronized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Iterator
import pickle


# ---------------------------------------------------------------------
# Layer report
# ---------------------------------------------------------------------


@dataclass
class LayerReport:
    """
    Benchmark statistics for one intermediate layer.
    """

    layer: str
    matrix: object
    score: float
    integral: float


# ---------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------


@dataclass
class Snapshot:
    """
    One complete benchmark evaluation.

    Usually corresponds to one trained network at one point in time.
    """

    # ---------------------------------------------------------
    # Canonical experiment identity
    # ---------------------------------------------------------

    model: Optional[str] = None
    dataset: Optional[str] = None
    trainer: Optional[str] = None
    run: Optional[str] = None
    epoch: Optional[int] = None

    # ---------------------------------------------------------
    # Benchmark statistics
    # ---------------------------------------------------------

    accuracy: Optional[float] = None

    fgsm_accuracy: Optional[float] = None
    pgd_accuracy: Optional[float] = None

    interpretability_score: Optional[float] = None
    interpretability_integral: Optional[float] = None

    layers: List[LayerReport] = field(default_factory=list)

    # ---------------------------------------------------------
    # Legacy / extensible metadata
    # ---------------------------------------------------------

    metadata: dict = field(default_factory=dict)

    # ---------------------------------------------------------

    def __post_init__(self):
        """
        Synchronize canonical metadata with the legacy metadata
        dictionary.

        This preserves backward compatibility with older report files
        and older producer code.
        """

        canonical = (
            "model",
            "dataset",
            "trainer",
            "run",
            "epoch",
        )

        #
        # New-style construction
        #
        # Snapshot(model="ResNet18", ...)
        #
        # populates metadata.
        #

        for key in canonical:
            value = getattr(self, key)
            if value is not None:
                self.metadata.setdefault(key, value)

        #
        # Old-style construction
        #
        # Snapshot(metadata={"model": ...})
        #
        # populates canonical attributes.
        #

        for key in canonical:
            if getattr(self, key) is None and key in self.metadata:
                setattr(self, key, self.metadata[key])


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------


class Report:
    """
    Canonical InterpBench report.

    Behaves like a list of Snapshot objects.

    A Report is intentionally simple: it is merely an ordered collection
    of Snapshots. One Report may contain one Snapshot, one training
    history, or multiple histories. Consumers reconstruct histories
    using the canonical Snapshot metadata
    (model, dataset, trainer, run, epoch).
    """

    def __init__(self):
        self.snapshots: List[Snapshot] = []

    # ---------------------------------------------------------

    def add(self, snapshot: Snapshot):
        """
        Append one Snapshot.
        """
        self.snapshots.append(snapshot)

    # ---------------------------------------------------------

    def extend(self, other: "Report"):
        """
        Append all snapshots from another Report.
        """
        self.snapshots.extend(other.snapshots)

    # ---------------------------------------------------------

    def clear(self):
        self.snapshots.clear()

    # ---------------------------------------------------------

    def save(self, filename):
        """
        Serialize report to disk.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    # ---------------------------------------------------------

    @classmethod
    def load(cls, filename):
        """
        Load report from disk.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    # ---------------------------------------------------------

    def copy(self):
        """
        Deep-copy the report.
        """
        import copy

        return copy.deepcopy(self)

    # ---------------------------------------------------------

    @property
    def latest(self):
        """
        Return the newest Snapshot.
        """
        if not self.snapshots:
            return None
        return self.snapshots[-1]

    # ---------------------------------------------------------

    def histories(self):
        """
        Iterate over logical histories contained in the Report.

        Returns
        -------
        dict
            Mapping

                (model, dataset, trainer, run)

            -> ordered list of Snapshots sorted by epoch.
        """

        from collections import defaultdict

        histories = defaultdict(list)

        for snapshot in self.snapshots:

            key = (
                snapshot.model,
                snapshot.dataset,
                snapshot.trainer,
                snapshot.run,
            )

            histories[key].append(snapshot)

        for history in histories.values():
            history.sort(
                key=lambda s: (
                    -1 if s.epoch is None else s.epoch
                )
            )

        return dict(histories)

    # ---------------------------------------------------------

    def __len__(self):
        return len(self.snapshots)

    # ---------------------------------------------------------

    def __getitem__(self, idx):
        return self.snapshots[idx]

    # ---------------------------------------------------------

    def __iter__(self) -> Iterator[Snapshot]:
        return iter(self.snapshots)

    # ---------------------------------------------------------

    def __repr__(self):

        if len(self) == 0:
            return "Report(0 snapshots)"

        histories = self.histories()

        last = self.latest

        return (
            f"Report("
            f"{len(self)} snapshots, "
            f"{len(histories)} histories, "
            f"accuracy={last.accuracy}, "
            f"fgsm={last.fgsm_accuracy}, "
            f"pgd={last.pgd_accuracy}, "
            f"score={last.interpretability_score}, "
            f"integral={last.interpretability_integral})"
        )