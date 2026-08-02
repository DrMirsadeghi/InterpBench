"""
plot.py

Visualization engine for InterpBench.

Public API
----------

>>> import interpbench as ib
>>> ib.plot(report)

The plotting routine automatically adapts to the Report
structure and automatically saves each figure as

    <model>_<dataset>_<trainer>_<date>.png

using the same text as the figure title.
"""

from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _metadata(obj, name, default):
    """
    Read metadata from an object.

    Supports both attributes and dictionary-like objects.
    """

    if hasattr(obj, name):
        value = getattr(obj, name)
        if value is not None:
            return value

    if isinstance(obj, dict):
        return obj.get(name, default)

    return default


def _figure_name(obj):
    """
    Construct figure title and filename.

    Returns
    -------
    title : str
    filename : str
    """

    model = _metadata(obj, "model", "UnknownModel")
    dataset = _metadata(obj, "dataset", "UnknownDataset")
    trainer = _metadata(obj, "trainer", "UnknownTrainer")
    date = _metadata(obj, "date", "UnknownDate")

    title = f"{model}_{dataset}_{trainer}_{date}"

    filename = re.sub(r"[^\w.-]+", "_", title) + ".png"

    return title, filename


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def plot(report):
    """
    Visualize an InterpBench Report.

    Parameters
    ----------
    report : Report
    """

    if len(report) == 0:
        raise ValueError("Empty Report.")

    if len(report) == 1:
        _plot_snapshot(report[0])
    else:
        _plot_history(report.histories())

    plt.show()


# ---------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------


def _plot_snapshot(snapshot):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True,
    )

    title, filename = _figure_name(snapshot)
    fig.suptitle(title, fontsize=16)

    #
    # ---------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------
    #

    ax = axes[0, 0]

    labels = ["Standard", "FGSM", "PGD"]

    values = [
        snapshot.accuracy,
        snapshot.fgsm_accuracy,
        snapshot.pgd_accuracy,
    ]

    ax.bar(labels, values)

    ax.set_ylim(0, 100)

    ax.set_title("Classification Performance")
    ax.set_ylabel("Accuracy (%)")

    #
    # ---------------------------------------------------------
    # Layer Score
    # ---------------------------------------------------------
    #

    ax = axes[0, 1]

    scores = [layer.score for layer in snapshot.layers]

    ax.plot(scores, "-o")

    ax.set_title("Layer Interpretability Score")
    ax.set_xlabel("Layer")

    #
    # ---------------------------------------------------------
    # Layer Integral
    # ---------------------------------------------------------
    #

    ax = axes[1, 0]

    integrals = [layer.integral for layer in snapshot.layers]

    ax.plot(integrals, "-o")

    ax.set_title("Layer Interpretability Integral")
    ax.set_xlabel("Layer")

    #
    # ---------------------------------------------------------
    # Heatmap
    # ---------------------------------------------------------
    #

    ax = axes[1, 1]

    if len(snapshot.layers):

        im = ax.imshow(
            snapshot.layers[-1].matrix,
            cmap="coolwarm",
            interpolation="nearest",
        )

        plt.colorbar(im, ax=ax)

    ax.set_title("Final Layer Geometry")

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )


# ---------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------

def _plot_history(histories):
    """
    Plot training histories for all benchmark metrics.

    The figure summarizes the evolution of

        Accuracy
        FGSM Accuracy
        PGD Accuracy
        Interpretability Score
        Interpretability Integral

    for every experimental history contained in the Report.

    The resulting figure is saved as

        every_history.png
    """

    import numpy as np
    import matplotlib.pyplot as plt

    metrics = [
        ("accuracy", "Accuracy", "Accuracy (%)"),
        ("fgsm_accuracy", "FGSM Accuracy", "Accuracy (%)"),
        ("pgd_accuracy", "PGD Accuracy", "Accuracy (%)"),
        (
            "interpretability_score",
            "Interpretability Score",
            "Score",
        ),
        (
            "interpretability_integral",
            "Interpretability Integral",
            "Integral",
        ),
    ]

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=1,
        figsize=(12, 16),
        sharex=True,
        constrained_layout=True,
    )

    all_epochs = set()

    for key, history in histories.items():

        if not history:
            continue

        history = sorted(
            history,
            key=lambda s: (
                float("inf") if s.epoch is None else s.epoch
            ),
        )

        epochs = [
            s.epoch if s.epoch is not None else i
            for i, s in enumerate(history)
        ]

        all_epochs.update(epochs)

        label = str(key)

        for ax, (attribute, title, ylabel) in zip(
            axes,
            metrics,
        ):

            values = [
                getattr(s, attribute, np.nan)
                for s in history
            ]

            ax.plot(
                epochs,
                values,
                marker="o",
                linewidth=2,
                markersize=5,
                label=label,
            )

            ax.set_title(title, loc="left", fontsize=12)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    #
    # Shared epoch axis
    #

    if all_epochs:
        ticks = sorted(all_epochs)
        for ax in axes:
            ax.set_xticks(ticks)

    axes[-1].set_xlabel("Epoch")

    #
    # Accuracy panels use a common scale.
    #

    for ax in axes[:3]:
        ax.set_ylim(0, 100)

    #
    # Show legend only once.
    #

    if len(histories) > 1:
        axes[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.30),
            ncol=min(4, len(histories)),
            frameon=False,
        )

    fig.savefig(
        "every_history.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    #
    # Summary statistics
    #

    _report_summary_table(histories)
    _all_statistics(histories)

def _report_summary_table(histories):
    """
    Generate the benchmark summary table.

    Groups histories by

        (model, dataset, trainer)

    and reports the final epoch statistics as

        mean ± 95% confidence interval

    across runs.

    Outputs
    -------
    1. Pretty console table
    2. summary_table.csv
    3. summary_table.tex

    Notes
    -----
    * The final Snapshot from each run is used.
    * Confidence intervals are computed across runs.
    * If only one run exists, the confidence interval is reported as NaN.
    """

    import csv
    import math
    from collections import defaultdict

    import numpy as np

    try:
        from scipy.stats import t
        HAVE_SCIPY = True
    except Exception:
        HAVE_SCIPY = False

    # -------------------------------------------------------------
    # Group final snapshots by (model, dataset, trainer)
    # -------------------------------------------------------------

    grouped = defaultdict(list)

    for (model, dataset, trainer, run), history in histories.items():

        if len(history) == 0:
            continue

        #
        # Histories are already sorted by epoch.
        # Use the newest snapshot.
        #

        #latest = history[-1]
        latest = max(
            history,
            key=lambda s: (
                float("-inf") if s.epoch is None else s.epoch
            )
        )
        print("History key :", (model, dataset, trainer, run))
        print("Snapshot key:", (
            latest.model,
            latest.dataset,
            latest.trainer,
            latest.run,
        ))
        print()

        grouped[(model, dataset, trainer)].append(latest)

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------

    def mean_ci(values):
        """
        Return

            mean, ci95

        where ci95 is NaN if fewer than two observations exist.
        """

        values = [
            float(v)
            for v in values
            if v is not None and not np.isnan(v)
        ]

        if len(values) == 0:
            return np.nan, np.nan

        mean = float(np.mean(values))

        #
        # Cannot estimate run-to-run uncertainty
        #

        if len(values) == 1:
            return mean, np.nan

        std = float(np.std(values, ddof=1))
        sem = std / math.sqrt(len(values))

        #
        # Student-t interval
        #

        if HAVE_SCIPY:
            crit = t.ppf(0.975, len(values) - 1)
        else:
            #
            # Conservative normal approximation if SciPy
            # is unavailable.
            #
            crit = 1.96

        ci = crit * sem

        return mean, ci

    def fmt(mean, ci, precision=4):
        if np.isnan(mean):
            return "N/A"

        if np.isnan(ci):
            return f"{mean:.{precision}f} ± N/A"

        return f"{mean:.{precision}f} ± {ci:.{precision}f}"

    # -------------------------------------------------------------
    # Build rows
    # -------------------------------------------------------------

    rows = []

    for key in sorted(grouped.keys()):

        model, dataset, trainer = key

        snapshots = grouped[key]

        accuracy = [
            s.accuracy
            for s in snapshots
        ]

        pgd = [
            s.pgd_accuracy
            for s in snapshots
        ]

        score = [
            s.interpretability_score
            for s in snapshots
        ]

        integral = [
            s.interpretability_integral
            for s in snapshots
        ]

        f1_robust = []

        for s in snapshots:
            if (
                s.accuracy is None
                or s.pgd_accuracy is None
                or (s.accuracy + s.pgd_accuracy) == 0
            ):
                f1_robust.append(np.nan)
            else:
                f1_robust.append(
                    2.0
                    * s.accuracy
                    * s.pgd_accuracy
                    / (s.accuracy + s.pgd_accuracy)
                )

        f1_mean, f1_ci = mean_ci(f1_robust)
        acc_mean, acc_ci = mean_ci(accuracy)
        pgd_mean, pgd_ci = mean_ci(pgd)
        
        score_mean, score_ci = mean_ci(score)
        integ_mean, integ_ci = mean_ci(integral)

        rows.append(
            dict(
                model=model,
                dataset=dataset,
                trainer=trainer,

                accuracy_mean=acc_mean,
                accuracy_ci=acc_ci,

                pgd_mean=pgd_mean,
                pgd_ci=pgd_ci,

                f1_mean=f1_mean,
                f1_ci=f1_ci,

                score_mean=score_mean,
                score_ci=score_ci,

                integral_mean=integ_mean,
                integral_ci=integ_ci,
            )
        )

    # -------------------------------------------------------------
    # Console table
    # -------------------------------------------------------------

    headers = [
        "Model",
        "Dataset",
        "Trainer",
        "Accuracy",
        "PGD",
        "F1-Robust",
        "Score",
        "Integral",
    ]

    widths = [
        18,
        16,
        16,
        22,
        22,
        22,
        22,
        22,
    ]

    line = "".join(
        h.ljust(w)
        for h, w in zip(headers, widths)
    )

    print()
    print("=" * len(line))
    print(line)
    print("=" * len(line))

    for r in rows:

        values = [

            r["model"],

            r["dataset"],

            r["trainer"],

            fmt(
                r["accuracy_mean"],
                r["accuracy_ci"],
            ),

            fmt(
                r["pgd_mean"],
                r["pgd_ci"],
            ),

            fmt(
                r["f1_mean"],
                r["f1_ci"],
            ),

            fmt(
                r["score_mean"],
                r["score_ci"],
            ),

            fmt(
                r["integral_mean"],
                r["integral_ci"],
            ),
        ]

        print(
            "".join(
                str(v).ljust(w)
                for v, w in zip(values, widths)
            )
        )

    print("=" * len(line))
    print()

    # -------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------

    csv_file = "summary_table.csv"

    with open(csv_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Model",
            "Dataset",
            "Trainer",

            "Accuracy Mean",
            "Accuracy CI95",

            "PGD Mean",
            "PGD CI95",

            "F1-Robust Mean",
            "F1-Robust CI95",

            "Interpretability Score Mean",
            "Interpretability Score CI95",

            "Interpretability Integral Mean",
            "Interpretability Integral CI95",
        ])

        for r in rows:

            writer.writerow([

                r["model"],
                r["dataset"],
                r["trainer"],

                r["accuracy_mean"],
                r["accuracy_ci"],

                r["pgd_mean"],
                r["pgd_ci"],

                r["f1_mean"],
                r["f1_ci"],

                r["score_mean"],
                r["score_ci"],

                r["integral_mean"],
                r["integral_ci"],
            ])

    # -------------------------------------------------------------
    # LaTeX
    # -------------------------------------------------------------

    tex_file = "summary_table.tex"

    with open(tex_file, "w") as f:

        f.write("\\begin{tabular}{lllccccc}\n")
        f.write("\\hline\n")

        f.write(
            "Model & Dataset & Trainer & "
            "Accuracy & PGD & F1-Robust & Score & Integral\\\\\n"
        )

        f.write("\\hline\n")

        for r in rows:

            f.write(

                f"{r['model']} & "
                f"{r['dataset']} & "
                f"{r['trainer']} & "
                f"{fmt(r['accuracy_mean'], r['accuracy_ci'])} & "
                f"{fmt(r['pgd_mean'], r['pgd_ci'])} & "
                f"{fmt(r['f1_mean'], r['f1_ci'])} & "
                f"{fmt(r['score_mean'], r['score_ci'])} & "
                f"{fmt(r['integral_mean'], r['integral_ci'])}"
                "\\\\\n"

            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    print(f"Saved summary table to {csv_file}")
    print(f"Saved LaTeX table to {tex_file}")

def _all_statistics(histories):
    """
    Generate benchmark summary statistics for every epoch.

    Histories are grouped by

        (model, dataset, trainer, epoch)

    and statistics are computed across runs for each epoch.

    Outputs
    -------
    1. Pretty console table
    2. all_statistics.csv
    3. all_statistics.tex
    """

    import csv
    import math
    from collections import defaultdict

    import numpy as np

    try:
        from scipy.stats import t
        HAVE_SCIPY = True
    except Exception:
        HAVE_SCIPY = False

    # -------------------------------------------------------------
    # Group snapshots by (model, dataset, trainer, epoch)
    # -------------------------------------------------------------

    grouped = defaultdict(list)

    for (model, dataset, trainer, run), history in histories.items():

        for snapshot in history:

            grouped[
                (
                    model,
                    dataset,
                    trainer,
                    snapshot.epoch,
                )
            ].append(snapshot)

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------

    def mean_ci(values):

        values = [
            float(v)
            for v in values
            if v is not None and not np.isnan(v)
        ]

        if len(values) == 0:
            return np.nan, np.nan

        mean = float(np.mean(values))

        if len(values) == 1:
            return mean, np.nan

        std = float(np.std(values, ddof=1))
        sem = std / math.sqrt(len(values))

        if HAVE_SCIPY:
            crit = t.ppf(0.975, len(values) - 1)
        else:
            crit = 1.96

        return mean, crit * sem

    def fmt(mean, ci, precision=4):

        if np.isnan(mean):
            return "N/A"

        if np.isnan(ci):
            return f"{mean:.{precision}f} ± N/A"

        return f"{mean:.{precision}f} ± {ci:.{precision}f}"

    # -------------------------------------------------------------
    # Build rows
    # -------------------------------------------------------------

    rows = []

    for key in sorted(grouped.keys()):

        model, dataset, trainer, epoch = key

        snapshots = grouped[key]

        accuracy = [s.accuracy for s in snapshots]
        pgd = [s.pgd_accuracy for s in snapshots]
        score = [s.interpretability_score for s in snapshots]
        integral = [s.interpretability_integral for s in snapshots]

        f1_robust = []

        for s in snapshots:
            if (
                s.accuracy is None
                or s.pgd_accuracy is None
                or (s.accuracy + s.pgd_accuracy) == 0
            ):
                f1_robust.append(np.nan)
            else:
                f1_robust.append(
                    2.0
                    * s.accuracy
                    * s.pgd_accuracy
                    / (s.accuracy + s.pgd_accuracy)
                )

        f1_mean, f1_ci = mean_ci(f1_robust)
        acc_mean, acc_ci = mean_ci(accuracy)
        pgd_mean, pgd_ci = mean_ci(pgd)
        
        score_mean, score_ci = mean_ci(score)
        integ_mean, integ_ci = mean_ci(integral)

        rows.append(
            dict(
                model=model,
                dataset=dataset,
                trainer=trainer,
                epoch=epoch,
                accuracy_mean=acc_mean,
                accuracy_ci=acc_ci,
                pgd_mean=pgd_mean,
                pgd_ci=pgd_ci,
                f1_mean=f1_mean,
                f1_ci=f1_ci,
                score_mean=score_mean,
                score_ci=score_ci,
                integral_mean=integ_mean,
                integral_ci=integ_ci,
            )
        )

    # -------------------------------------------------------------
    # Console table
    # -------------------------------------------------------------

    headers = [
        "Model",
        "Dataset",
        "Trainer",
        "Epoch",
        "Accuracy",
        "PGD",
        "F1-Robust",
        "Score",
        "Integral",
    ]

    widths = [
        18,
        16,
        16,
        8,
        22,
        22,
        22,
        22,
        22,
    ]

    line = "".join(
        h.ljust(w)
        for h, w in zip(headers, widths)
    )

    print()
    print("=" * len(line))
    print(line)
    print("=" * len(line))

    for r in rows:

        values = [
            r["model"],
            r["dataset"],
            r["trainer"],
            r["epoch"],
            fmt(r["accuracy_mean"], r["accuracy_ci"]),
            fmt(r["pgd_mean"], r["pgd_ci"]),
            fmt(r["f1_mean"], r["f1_ci"]),
            fmt(r["score_mean"], r["score_ci"]),
            fmt(r["integral_mean"], r["integral_ci"]),
        ]

        print(
            "".join(
                str(v).ljust(w)
                for v, w in zip(values, widths)
            )
        )

    print("=" * len(line))
    print()

    # -------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------

    csv_file = "all_statistics.csv"

    with open(csv_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Model",
            "Dataset",
            "Trainer",
            "Epoch",
            "Accuracy Mean",
            "Accuracy CI95",
            "PGD Mean",
            "PGD CI95",
            "F1-Robust Mean",
            "F1-Robust CI95",
            "Interpretability Score Mean",
            "Interpretability Score CI95",
            "Interpretability Integral Mean",
            "Interpretability Integral CI95",
        ])

        for r in rows:

            writer.writerow([
                r["model"],
                r["dataset"],
                r["trainer"],
                r["epoch"],
                r["accuracy_mean"],
                r["accuracy_ci"],
                r["pgd_mean"],
                r["pgd_ci"],
                r["f1_mean"],
                r["f1_ci"],
                r["score_mean"],
                r["score_ci"],
                r["integral_mean"],
                r["integral_ci"],
            ])

    # -------------------------------------------------------------
    # LaTeX
    # -------------------------------------------------------------

    tex_file = "all_statistics.tex"

    with open(tex_file, "w") as f:

        f.write("\\begin{tabular}{llllccccc}\n")
        f.write("\\hline\n")

        f.write(
            "Model & Dataset & Trainer & Epoch & "
            "Accuracy & PGD & F1-Robust & Score & Integral\\\\\n"
        )

        f.write("\\hline\n")

        for r in rows:

            f.write(
                f"{r['model']} & "
                f"{r['dataset']} & "
                f"{r['trainer']} & "
                f"{r['epoch']} & "
                f"{fmt(r['accuracy_mean'], r['accuracy_ci'])} & "
                f"{fmt(r['pgd_mean'], r['pgd_ci'])} & "
                f"{fmt(r['f1_mean'], r['f1_ci'])} & "
                f"{fmt(r['score_mean'], r['score_ci'])} & "
                f"{fmt(r['integral_mean'], r['integral_ci'])}"
                "\\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    print(f"Saved all statistics to {csv_file}")
    print(f"Saved LaTeX table to {tex_file}")