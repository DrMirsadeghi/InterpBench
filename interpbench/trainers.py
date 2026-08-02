"""
trainers.py

Training interfaces for InterpBench.

The trainer API mirrors the philosophy of the library:

    model
        ↓
    training
        ↓
    periodic evaluate(...)
        ↓
    append Snapshot
        ↓
    Report

The benchmark implementation itself lives entirely inside
evaluate.py. Trainers simply invoke evaluate() and accumulate the
returned snapshots.

Public classes
--------------

StandardTrainer
PGDTrainer
"""

from copy import deepcopy

from .evaluate import evaluate
from .report import Report

from mair.defenses.advtraining.standard import Standard
from mair.defenses.advtraining.at import AT


# ---------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------

class _InterpBenchMixin:
    """
    Shared functionality for all InterpBench trainers.

    Every benchmark checkpoint is appended to one Report.
    """

    def __init__(self, model, trainloader, testloader):

        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

        self.report = Report()

    # ---------------------------------------------------------

    def benchmark(self):
        """
        Evaluate current model and append the resulting Snapshot.
        """

        current = evaluate(
            self.model,
            self.testloader,
            model_name=self.model_name,
            dataset=self.dataset_name,
            trainer=self.training_method,
            run=self.run,
            epoch=self.current_epoch,
        )

        self.report.extend(current)

    # ---------------------------------------------------------
    def fit(
        self,
        epochs=1,
        *,
        benchmark_every=1,
        model=None,
        dataset=None,
        trainer=None,
        run=None,
    ):
        """
        Train while periodically benchmarking.

        Parameters
        ----------
        epochs : int

        model : str, optional
            Model name.

        dataset : str, optional
            Dataset name.

        trainer : str, optional
            Training method (e.g. STANDARD, PGD).

        run : str, optional
            Run identifier.
        """
        self.benchmark_every = benchmark_every
        self.model_name = model
        self.dataset_name = dataset
        self.training_method = trainer
        self.run = run

        #
        # Initial network
        #

        self.current_epoch = 0
        self.benchmark()

        #
        # Training
        #

        self._fit(epochs)

        return self.report
# ---------------------------------------------------------------------
# Standard training
# ---------------------------------------------------------------------


class StandardTrainer(_InterpBenchMixin):
    """
    Standard empirical-risk-minimization training.

    Parameters
    ----------
    model : mair.RobModel

    trainloader : DataLoader

    testloader : DataLoader
    """

    def __init__(
        self,
        model,
        trainloader,
        testloader,
    ):

        super().__init__(
            model,
            trainloader,
            testloader,
        )

        self.trainer = Standard(model)

    # ---------------------------------------------------------

    def _fit(self, epochs):

        self.trainer.record_rob(
            self.trainloader,
            self.testloader,
            eps=8 / 255,
            alpha=2 / 255,
            steps=10,
        )

        self.trainer.setup(
            optimizer="SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)",
            scheduler="Step(milestones=[100,150], gamma=0.1)",
            scheduler_type="Epoch",
            minimizer=None,
            n_epochs=epochs,
            n_iters=len(self.trainloader),
        )

        #
        # Benchmark every epoch
        #

        for epoch in range(1, epochs + 1):

            self.trainer.fit(
                train_loader=self.trainloader,
                n_epochs=1,
                save_path=None,
                save_best=None,
                save_type=None,
                save_overwrite=True,
                record_type="Epoch",
            )
            self.current_epoch = epoch
            if (
                epoch % self.benchmark_every == 0
                or epoch == epochs
            ):
                self.benchmark()


# ---------------------------------------------------------------------
# PGD adversarial training
# ---------------------------------------------------------------------


class PGDTrainer(_InterpBenchMixin):
    """
    PGD adversarial training.
    """

    def __init__(
        self,
        model,
        trainloader,
        testloader,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
    ):

        super().__init__(
            model,
            trainloader,
            testloader,
        )

        self.trainer = AT(
            model,
            eps=eps,
            alpha=alpha,
            steps=steps,
        )

    # ---------------------------------------------------------

    def _fit(self, epochs):

        self.trainer.record_rob(
            self.trainloader,
            self.testloader,
            eps=8 / 255,
            alpha=2 / 255,
            steps=10,
        )

        self.trainer.setup(
            optimizer="SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)",
            scheduler="Step(milestones=[100,150], gamma=0.1)",
            scheduler_type="Epoch",
            minimizer=None,
            n_epochs=epochs,
            n_iters=len(self.trainloader),
        )
        for epoch in range(1, epochs + 1):

            self.trainer.fit(
                train_loader=self.trainloader,
                n_epochs=1,
                save_path=None,
                save_best=None,
                save_type=None,
                save_overwrite=True,
                record_type="Epoch",
            )

            self.current_epoch = epoch
            if (
                epoch % self.benchmark_every == 0
                or epoch == epochs
            ):
                self.benchmark()        