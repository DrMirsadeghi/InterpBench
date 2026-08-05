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

import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _make_title_page(histories) -> str:
    """
    Generate title page metadata.
    No analysis is performed here.
    """

    models = sorted(
        {
            h.model
            for h in histories
            if getattr(h, "model", None) is not None
        }
    )

    datasets = sorted(
        {
            h.dataset
            for h in histories
            if getattr(h, "dataset", None) is not None
        }
    )

    trainers = sorted(
        {
            h.trainer
            for h in histories
            if getattr(h, "trainer", None) is not None
        }
    )

    runs = sorted(
        {
            h.run
            for h in histories
            if getattr(h, "run", None) is not None
        }
    )

    epochs = sorted(
        {
            h.epoch
            for h in histories
            if getattr(h, "epoch", None) is not None
        }
    )

    lines = [
        r"\begin{tabular}{ll}",
        r"\textbf{Generated:} & "
        + datetime.now().strftime("%Y-%m-%d %H:%M"),
        r"\\",
        r"\textbf{Models:} & "
        + ", ".join(map(str, models)),
        r"\\",
        r"\textbf{Datasets:} & "
        + ", ".join(map(str, datasets)),
        r"\\",
        r"\textbf{Trainers:} & "
        + ", ".join(map(str, trainers)),
        r"\\",
        r"\textbf{Runs:} & "
        + ", ".join(map(str, runs)),
        r"\\",
    ]

    if epochs:
        lines.append(
            r"\textbf{Epoch range:} & "
            f"{min(epochs)}--{max(epochs)}"
        )
        lines.append(r"\\")

    lines.append(
        r"\end{tabular}"
    )

    return "\n".join(lines)

def _make_tables_section(output_dir: Path) -> str:
    """
    Generate landscape table sections.

    Tables are placed on landscape pages.
    Landscape pages have no header/footer/page numbering.
    Tables are automatically scaled to fit the available width.
    """

    tables = [
        ("Summary", "summary_table.tex"),
        ("Benchmark Comparison", "table_2.tex"),
        ("Overall Statistics", "all_statistics.tex"),
        ("Significance Tests", "significance_table.tex"),
    ]

    content = []

    existing_tables = [
        (title, filename)
        for title, filename in tables
        if (output_dir / filename).exists()
    ]

    if not existing_tables:
        return ""

    content.append(
        r"""
\clearpage

\begin{landscape}

% Remove header/footer/page numbering on landscape pages
\pagestyle{empty}

\section{Tables}
"""
    )

    for title, filename in existing_tables:

        content.append(
            rf"""
\subsection{{{title}}}

\begin{{table}}[H]
\centering
\small
\resizebox{{\linewidth}}{{!}}{{%
    \input{{{filename}}}
}}
\end{{table}}

\clearpage
"""
        )

    content.append(
        r"""
\end{landscape}

% Restore normal portrait page style
\pagestyle{fancy}

\clearpage
"""
    )

    return "\n".join(content)

def _make_figures_section(output_dir: Path) -> str:

    figures = [

        # -------------------------------------------------
        # Existing benchmark figures
        # -------------------------------------------------

        ("Training Curves",
         [
             "figure_1_training_curves.png",
         ]),

        ("Stability",
         [
             "figure_2_stability.png",
         ]),

        ("Best vs Last Checkpoints",
         [
             "figure_3_best_last_checkpoints.png",
         ]),

        ("Trade-off Analysis",
         [
             "figure_3_best_trade_offs.png",
             "figure_3_last_trade_offs.png",
             "figure_3_all_trade_offs.png",
         ]),

        ("Distribution Analysis",
         [
             "figure_4_distros.png",
             "figure_4_distros_last.png",
             "figure_4_distros_all.png",
         ]),

        ("Layer-wise Scores",
         [
             "figure_5_layerwise_score.png",
             "figure_5_layerwise_score_last.png",
             "figure_5_layerwise_score_all.png",
         ]),


        # -------------------------------------------------
        # Extended interpretability figures
        # -------------------------------------------------

        ("Pairwise Relationship Matrix",
         [
             "figure_6_pairwise_relationship_matrix.png",
         ]),

        ("Best vs Last Interpretability",
         [
             "figure_7_best_last.png",
         ]),

        ("Layer-wise Matrix Analysis",
         [
             "figure_8_layerwise_matrix.png",
         ]),

        ("Width-Integral Relationship",
         [
             "figure_9_width_integral_relationship.png",
         ]),

        ("Layer-wise Score Profile",
         [
             "figure_10_layerwise_score_profile.png",
         ]),

        ("Layer-Network Aggregation Consistency",
         [
             "figure_11_layer_network_aggregation_check.png",
         ]),

        ("Layer Importance / Contribution Ranking",
         [
             "figure_12_layer_importance_contribution.png",
         ]),
        (
            "Layer-wise Score by Training Regime",
            [
                "figure_12_layerwise_score_by_trainer.png",
            ],
        ),
    ]


    content = []

    for title, files in figures:

        existing = [
            f for f in files
            if (output_dir / f).exists()
        ]

        if not existing:
            continue

        content.append(
            rf"""
\subsection{{{title}}}
"""
        )

        for filename in existing:
            content.append(
                rf"""
\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{filename}}}
\end{{figure}}
"""
            )

    return "\n".join(content)

def _save_pdf(
    histories,
    output_dir,
    filename="InterpBench_Report.pdf",
):
    """
    Assemble generated tables and figures into a LaTeX report.

    If lualatex is available:
        - compile the report into PDF
        - remove auxiliary files
        - return PDF path

    If lualatex is unavailable:
        - save the generated .tex file
        - return .tex path
    """

    import shutil
    import subprocess
    import warnings
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    template_path = (
        Path(__file__).with_name("report.tex")
    )

    if not template_path.exists():
        raise FileNotFoundError(
            f"Missing LaTeX template: {template_path}"
        )

    # --------------------------------------------------
    # Load template
    # --------------------------------------------------

    with open(
        template_path,
        "r",
        encoding="utf-8",
    ) as f:
        template = f.read()

    # --------------------------------------------------
    # Fill placeholders
    # --------------------------------------------------

    template = template.replace(
        "%%TITLE_PAGE%%",
        _make_title_page(histories),
    )

    template = template.replace(
        "%%TABLES_SECTION%%",
        _make_tables_section(output_dir),
    )

    template = template.replace(
        "%%FIGURES_SECTION%%",
        _make_figures_section(output_dir),
    )

    # --------------------------------------------------
    # Write temporary LaTeX document
    # --------------------------------------------------

    tex_path = output_dir / (
        Path(filename).stem + ".tex"
    )

    with open(
        tex_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(template)

    # --------------------------------------------------
    # Check LaTeX availability
    # --------------------------------------------------

    compiler = shutil.which("lualatex")

    if compiler is None:
        warnings.warn(
            "lualatex was not found. "
            "Returning the generated LaTeX file instead "
            "of compiling the PDF.",
            RuntimeWarning,
        )

        return tex_path

    # --------------------------------------------------
    # Compile PDF
    # --------------------------------------------------

    try:

        for _ in range(2):
            subprocess.run(
                [
                    compiler,
                    "-interaction=nonstopmode",
                    tex_path.name,
                ],
                cwd=output_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

    except subprocess.CalledProcessError as exc:

        warnings.warn(
            "LaTeX compilation failed. "
            "Returning the generated .tex file.\n"
            f"{exc}",
            RuntimeWarning,
        )

        return tex_path

    # --------------------------------------------------
    # Remove auxiliary files
    # --------------------------------------------------

    for ext in [
        ".aux",
        ".log",
        ".toc",
        ".out",
    ]:
        aux = tex_path.with_suffix(ext)

        if aux.exists():
            aux.unlink()

    # --------------------------------------------------
    # Return PDF
    # --------------------------------------------------

    pdf_path = tex_path.with_suffix(".pdf")

    if pdf_path.exists():

        final_pdf = output_dir / filename

        if pdf_path != final_pdf:
            shutil.move(
                str(pdf_path),
                str(final_pdf),
            )

        return final_pdf

    # Unexpected case
    return tex_path

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
    _report_table_2(histories)
    _all_statistics(histories)
    _report_significance_test(histories)
    _figure_1_training_curves(histories)
    _figure_3_best_last_checkpoints(histories)
    _figure_2_stability(histories)
    _figure_III_trade_offs(histories)
    _figure_III_trade_offs(histories,checkpoint="last")
    _figure_III_trade_offs(histories,checkpoint="all")
    _figure_4_distros(histories,checkpoint="best")
    _figure_4_distros(histories,checkpoint="last")
    _figure_5_layerwise_score(histories)
    _figure_5_layerwise_score(histories,checkpoint="last")
    _figure_5_layerwise_score(histories,checkpoint="all")

    _plot_pairwise_relationship_matrix(histories)
    _figure_7_best_last(histories)
    _figure_8_layerwise_matrix(histories)
    _figure_9_width_integral_relationship(histories)
    _figure_10_layerwise_score_profile(histories)
    _figure_11_layer_network_aggregation_check(histories)
    _save_pdf(histories,"./","InterpBench_Report.pdf")

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
    Generate hierarchical benchmark statistics.

    Layout:

        model
        dataset
        trainer
        epoch
        run
        level

    where

        run = AVE
            -> mean ± 95% CI across runs

        run = integer
            -> raw observation

    Outputs
    -------
    all_statistics.csv
    all_statistics.tex
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
            crit = t.ppf(
                0.975,
                len(values)-1,
            )
        else:
            crit = 1.96

        return mean, crit * sem


    def fmt(mean, ci, scientific=False):

        if mean is None or np.isnan(mean):
            return ""

        if np.isnan(ci):
            if scientific:
                return f"{mean:.3e}"
            return f"{mean:.4f}"

        if scientific:
            return f"{mean:.3e}±{ci:.3e}"

        return f"{mean:.4f}±{ci:.4f}"


    def run_key(run):
        """
        Sorting rule:

            AVE first
            then numerical runs
        """

        if run == "AVE":
            return (0, -1)

        try:
            return (1, int(run))
        except Exception:
            return (2, str(run))


    # -------------------------------------------------------------
    # Group snapshots
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


    rows = []


    # -------------------------------------------------------------
    # Build hierarchical rows
    # -------------------------------------------------------------

    for (
        model,
        dataset,
        trainer,
        epoch,
    ) in sorted(grouped.keys()):


        snapshots = grouped[
            (
                model,
                dataset,
                trainer,
                epoch,
            )
        ]


        # ---------------------------------------------------------
        # AVE row
        # ---------------------------------------------------------

        def make_metrics(items):

            accuracy = [
                s.accuracy for s in items
            ]

            pgd = [
                s.pgd_accuracy for s in items
            ]

            score = [
                s.interpretability_score for s in items
            ]

            integral = [
                s.interpretability_integral for s in items
            ]

            f1 = []

            for s in items:

                if (
                    s.accuracy is None
                    or s.pgd_accuracy is None
                    or s.accuracy+s.pgd_accuracy == 0
                ):
                    f1.append(np.nan)

                else:
                    f1.append(
                        2*s.accuracy*s.pgd_accuracy /
                        (s.accuracy+s.pgd_accuracy)
                    )

            return {
                "accuracy": mean_ci(accuracy),
                "pgd": mean_ci(pgd),
                "f1": mean_ci(f1),
                "score": mean_ci(score),
                "integral": mean_ci(integral),
            }


        ave = make_metrics(snapshots)


        rows.append(
            {
                "model":model,
                "dataset":dataset,
                "trainer":trainer,
                "epoch":epoch,
                "run":"AVE",
                "level":"Overall",

                "accuracy_mean":ave["accuracy"][0],
                "accuracy_ci":ave["accuracy"][1],

                "pgd_mean":ave["pgd"][0],
                "pgd_ci":ave["pgd"][1],

                "f1_mean":ave["f1"][0],
                "f1_ci":ave["f1"][1],

                "score_mean":ave["score"][0],
                "score_ci":ave["score"][1],

                "integral_mean":ave["integral"][0],
                "integral_ci":ave["integral"][1],
            }
        )


        # ---------------------------------------------------------
        # AVE layer rows
        # ---------------------------------------------------------

        max_layers=max(
            len(s.layers)
            for s in snapshots
        )

        for layer_idx in range(max_layers):

            values=[]

            for s in snapshots:

                if layer_idx < len(s.layers):
                    values.append(
                        s.layers[layer_idx].score
                    )


            m,c=mean_ci(values)


            rows.append(
                {
                    "model":"",
                    "dataset":"",
                    "trainer":"",
                    "epoch":"",
                    "run":"AVE",
                    "level":f"Layer {layer_idx}",

                    "accuracy_mean":np.nan,
                    "accuracy_ci":np.nan,

                    "pgd_mean":np.nan,
                    "pgd_ci":np.nan,

                    "f1_mean":np.nan,
                    "f1_ci":np.nan,

                    "score_mean":m,
                    "score_ci":c,

                    "integral_mean":np.nan,
                    "integral_ci":np.nan,
                }
            )


        # ---------------------------------------------------------
        # Individual runs
        # ---------------------------------------------------------

        for snapshot in sorted(
            snapshots,
            key=lambda s: run_key(s.run)
        ):


            rows.append(
                {
                    "model":model,
                    "dataset":dataset,
                    "trainer":trainer,
                    "epoch":epoch,
                    "run":snapshot.run,
                    "level":"Overall",

                    "accuracy_mean":snapshot.accuracy,
                    "accuracy_ci":np.nan,

                    "pgd_mean":snapshot.pgd_accuracy,
                    "pgd_ci":np.nan,

                    "f1_mean":(
                        2*snapshot.accuracy*
                        snapshot.pgd_accuracy /
                        (snapshot.accuracy+
                         snapshot.pgd_accuracy)
                        if snapshot.accuracy is not None
                        and snapshot.pgd_accuracy is not None
                        and snapshot.accuracy+
                           snapshot.pgd_accuracy !=0
                        else np.nan
                    ),

                    "f1_ci":np.nan,

                    "score_mean":
                        snapshot.interpretability_score,

                    "score_ci":np.nan,

                    "integral_mean":
                        snapshot.interpretability_integral,

                    "integral_ci":np.nan,
                }
            )


            for idx,layer in enumerate(snapshot.layers):

                rows.append(
                    {
                        "model":"",
                        "dataset":"",
                        "trainer":"",
                        "epoch":"",
                        "run":snapshot.run,
                        "level":f"Layer {idx}",

                        "accuracy_mean":np.nan,
                        "accuracy_ci":np.nan,

                        "pgd_mean":np.nan,
                        "pgd_ci":np.nan,

                        "f1_mean":np.nan,
                        "f1_ci":np.nan,

                        "score_mean":layer.score,
                        "score_ci":np.nan,

                        "integral_mean":np.nan,
                        "integral_ci":np.nan,
                    }
                )


    # -------------------------------------------------------------
    # Sort
    # -------------------------------------------------------------

    rows.sort(
        key=lambda r:(
            r["model"],
            r["dataset"],
            r["trainer"],
            r["epoch"],
            run_key(r["run"]),
            r["level"],
        )
    )


    # -------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------

    with open(
        "all_statistics.csv",
        "w",
        newline=""
    ) as f:

        writer=csv.writer(f)

        writer.writerow(
            [
                "Model",
                "Dataset",
                "Trainer",
                "Epoch",
                "Run",
                "Level",

                "Accuracy Mean",
                "Accuracy CI95",

                "PGD Mean",
                "PGD CI95",

                "F1 Mean",
                "F1 CI95",

                "Score Mean",
                "Score CI95",

                "Integral Mean",
                "Integral CI95",
            ]
        )

        for r in rows:

            writer.writerow(
                [
                    r[k]
                    for k in [
                        "model",
                        "dataset",
                        "trainer",
                        "epoch",
                        "run",
                        "level",

                        "accuracy_mean",
                        "accuracy_ci",

                        "pgd_mean",
                        "pgd_ci",

                        "f1_mean",
                        "f1_ci",

                        "score_mean",
                        "score_ci",

                        "integral_mean",
                        "integral_ci",
                    ]
                ]
            )


    print(
        "Saved all_statistics.csv"
    )

def _report_table_2(histories):
    """
    Generate Table 2.

    Best-checkpoint vs last-checkpoint comparison.

    Best checkpoint:
        epoch with maximum interpretability score S

    Last checkpoint:
        final training epoch

    Reports

        • mean ± 95% CI across runs (Run = AVE)
        • individual runs

    Outputs
    -------
    table_2.csv
    table_2.tex
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

    def fmt(mean, ci):

        if np.isnan(mean):
            return ""

        if np.isnan(ci):
            return f"{mean:.4f} ± N/A"

        return f"{mean:.4f} ± {ci:.4f}"

    # -------------------------------------------------------------
    # Collect per-run statistics
    # -------------------------------------------------------------

    grouped = defaultdict(list)

    for (model, dataset, trainer, run), history in histories.items():

        if not history:
            continue

        last = max(
            history,
            key=lambda s: (
                float("-inf")
                if s.epoch is None
                else s.epoch
            ),
        )

        best = max(
            history,
            key=lambda s: (
                float("-inf")
                if s.interpretability_score is None
                else s.interpretability_score
            ),
        )

        grouped[(model, dataset, trainer)].append(
            dict(
                run=run,

                acc_best=best.accuracy,
                acc_last=last.accuracy,

                pgd_best=best.pgd_accuracy,
                pgd_last=last.pgd_accuracy,

                s_best=best.interpretability_score,
                s_last=last.interpretability_score,
            )
        )

    # -------------------------------------------------------------
    # Build rows
    # -------------------------------------------------------------

    rows = []

    for model, dataset, trainer in sorted(grouped.keys()):

        runs = grouped[(model, dataset, trainer)]

        #
        # Average row
        #

        acc_best_mean, acc_best_ci = mean_ci(
            [r["acc_best"] for r in runs]
        )

        acc_last_mean, acc_last_ci = mean_ci(
            [r["acc_last"] for r in runs]
        )

        pgd_best_mean, pgd_best_ci = mean_ci(
            [r["pgd_best"] for r in runs]
        )

        pgd_last_mean, pgd_last_ci = mean_ci(
            [r["pgd_last"] for r in runs]
        )

        s_best_mean, s_best_ci = mean_ci(
            [r["s_best"] for r in runs]
        )

        s_last_mean, s_last_ci = mean_ci(
            [r["s_last"] for r in runs]
        )

        rows.append(
            dict(
                Model=model,
                Dataset=dataset,
                Trainer=trainer,
                Run="AVE",

                AccBest=acc_best_mean,
                AccBestCI=acc_best_ci,

                AccLast=acc_last_mean,
                AccLastCI=acc_last_ci,

                PGDBest=pgd_best_mean,
                PGDBestCI=pgd_best_ci,

                PGDLast=pgd_last_mean,
                PGDLastCI=pgd_last_ci,

                SBest=s_best_mean,
                SBestCI=s_best_ci,

                SLast=s_last_mean,
                SLastCI=s_last_ci,

                average=True,
            )
        )

        #
        # Individual runs
        #

        for r in sorted(runs, key=lambda x: x["run"]):

            rows.append(
                dict(
                    Model="",
                    Dataset="",
                    Trainer="",
                    Run=r["run"],

                    AccBest=r["acc_best"],
                    AccBestCI=np.nan,

                    AccLast=r["acc_last"],
                    AccLastCI=np.nan,

                    PGDBest=r["pgd_best"],
                    PGDBestCI=np.nan,

                    PGDLast=r["pgd_last"],
                    PGDLastCI=np.nan,

                    SBest=r["s_best"],
                    SBestCI=np.nan,

                    SLast=r["s_last"],
                    SLastCI=np.nan,

                    average=False,
                )
            )

    # -------------------------------------------------------------
    # Console
    # -------------------------------------------------------------

    headers = [
        "Model",
        "Dataset",
        "Trainer",
        "Run",
        "Acc Best",
        "Acc Last",
        "PGD Best",
        "PGD Last",
        "S Best",
        "S Last",
    ]

    widths = [
        18,
        16,
        16,
        8,
        18,
        18,
        18,
        18,
        18,
        18,
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

        if r["average"]:

            values = [
                r["Model"],
                r["Dataset"],
                r["Trainer"],
                r["Run"],
                fmt(r["AccBest"], r["AccBestCI"]),
                fmt(r["AccLast"], r["AccLastCI"]),
                fmt(r["PGDBest"], r["PGDBestCI"]),
                fmt(r["PGDLast"], r["PGDLastCI"]),
                fmt(r["SBest"], r["SBestCI"]),
                fmt(r["SLast"], r["SLastCI"]),
            ]

        else:

            values = [
                r["Model"],
                r["Dataset"],
                r["Trainer"],
                r["Run"],
                f"{r['AccBest']:.4f}",
                f"{r['AccLast']:.4f}",
                f"{r['PGDBest']:.4f}",
                f"{r['PGDLast']:.4f}",
                f"{r['SBest']:.4f}",
                f"{r['SLast']:.4f}",
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

    with open(
        "table_2.csv",
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Model",
            "Dataset",
            "Trainer",
            "Run",
            "Acc Best",
            "Acc Best CI95",
            "Acc Last",
            "Acc Last CI95",
            "PGD Best",
            "PGD Best CI95",
            "PGD Last",
            "PGD Last CI95",
            "S Best",
            "S Best CI95",
            "S Last",
            "S Last CI95",
        ])

        for r in rows:

            writer.writerow([
                r["Model"],
                r["Dataset"],
                r["Trainer"],
                r["Run"],
                r["AccBest"],
                r["AccBestCI"],
                r["AccLast"],
                r["AccLastCI"],
                r["PGDBest"],
                r["PGDBestCI"],
                r["PGDLast"],
                r["PGDLastCI"],
                r["SBest"],
                r["SBestCI"],
                r["SLast"],
                r["SLastCI"],
            ])

    # -------------------------------------------------------------
    # LaTeX
    # -------------------------------------------------------------

    with open("table_2.tex", "w") as f:

        f.write("\\begin{tabular}{llllcccccc}\n")
        f.write("\\hline\n")

        f.write(
            "Model & Dataset & Trainer & Run & "
            "Acc Best & Acc Last & "
            "PGD Best & PGD Last & "
            "S Best & S Last\\\\\n"
        )

        f.write("\\hline\n")

        for r in rows:

            if r["average"]:

                acc_best = fmt(r["AccBest"], r["AccBestCI"])
                acc_last = fmt(r["AccLast"], r["AccLastCI"])
                pgd_best = fmt(r["PGDBest"], r["PGDBestCI"])
                pgd_last = fmt(r["PGDLast"], r["PGDLastCI"])
                s_best = fmt(r["SBest"], r["SBestCI"])
                s_last = fmt(r["SLast"], r["SLastCI"])

            else:

                acc_best = f"{r['AccBest']:.4f}"
                acc_last = f"{r['AccLast']:.4f}"
                pgd_best = f"{r['PGDBest']:.4f}"
                pgd_last = f"{r['PGDLast']:.4f}"
                s_best = f"{r['SBest']:.4f}"
                s_last = f"{r['SLast']:.4f}"

            f.write(
                f"{r['Model']} & "
                f"{r['Dataset']} & "
                f"{r['Trainer']} & "
                f"{r['Run']} & "
                f"{acc_best} & "
                f"{acc_last} & "
                f"{pgd_best} & "
                f"{pgd_last} & "
                f"{s_best} & "
                f"{s_last}"
                "\\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    print("Saved table_2.csv")
    print("Saved table_2.tex")

def _figure_1_training_curves(histories):
    """
    Figure 1.

    Training curves across metrics and pairings.

    Rows
    ----
    (model, dataset) pairings.

    Columns
    -------
    Accuracy
    PGD Accuracy
    Interpretability Score
    Interpretability Integral

    Curves
    ------
    Blue  : Standard training
    Red   : Adversarial training

    Shaded regions indicate ±1 standard deviation across runs.

    Output
    ------
    figure_1_training_curves.png
    """

    from collections import defaultdict

    import numpy as np
    import matplotlib.pyplot as plt

    #
    # -------------------------------------------------------------
    # Organize snapshots
    #
    # pairing -> trainer -> epoch -> snapshots
    # -------------------------------------------------------------
    #

    grouped = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )

    for (model, dataset, trainer, run), history in histories.items():

        for snapshot in history:

            grouped[
                (model, dataset)
            ][trainer][snapshot.epoch].append(snapshot)

    #
    # -------------------------------------------------------------
    # Determine ordering
    # -------------------------------------------------------------
    #

    pairings = sorted(grouped.keys())

    metrics = [
        (
            "accuracy",
            "Accuracy",
            "Accuracy (%)",
        ),
        (
            "pgd_accuracy",
            "PGD Accuracy",
            "Accuracy (%)",
        ),
        (
            "interpretability_score",
            "S (Interpretability)",
            "Score",
        ),
        (
            "interpretability_integral",
            "Integral",
            "Integral",
        ),
    ]

    #
    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------
    #

    fig, axes = plt.subplots(
        len(pairings),
        len(metrics),
        figsize=(12, 2.1 * len(pairings)),
        sharex=True,
        constrained_layout=True,
    )

    #
    # Ensure 2-D indexing even for a single pairing.
    #

    axes = np.atleast_2d(axes)

    #
    # Trainer colors.
    #

    trainer_colors = {
        "Standard": "tab:blue",
        "PGD": "tab:red",
    }

    #
    # -------------------------------------------------------------
    # Draw curves
    # -------------------------------------------------------------
    #

    for row, pairing in enumerate(pairings):

        model, dataset = pairing

        #
        # Row label.
        #

        axes[row, 0].set_ylabel(
            f"{model}\n{dataset}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
            labelpad=28,
        )

        #
        # Metric columns.
        #

        for col, (attribute, title, ylabel) in enumerate(metrics):

            ax = axes[row, col]

            #
            # Column titles only once.
            #

            if row == 0:
                ax.set_title(title, fontsize=11)

            #
            # Plot each trainer.
            #

            for trainer, epoch_dict in grouped[pairing].items():

                epochs = sorted(epoch_dict.keys())

                mean = []
                std = []

                for epoch in epochs:

                    values = [
                        getattr(s, attribute, np.nan)
                        for s in epoch_dict[epoch]
                    ]

                    values = np.asarray(values, dtype=float)

                    values = values[
                        ~np.isnan(values)
                    ]

                    if len(values):

                        mean.append(values.mean())
                        std.append(values.std(ddof=0))

                    else:

                        mean.append(np.nan)
                        std.append(np.nan)

                mean = np.asarray(mean)
                std = np.asarray(std)

                #
                # Choose color.
                #

                color = trainer_colors.get(
                    trainer,
                    None,
                )

                #
                # Mean curve.
                #

                ax.plot(
                    epochs,
                    mean,
                    linewidth=1.8,
                    color=color,
                    label=trainer,
                )

                #
                # ±1 std band.
                #

                ax.fill_between(
                    epochs,
                    mean - std,
                    mean + std,
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                )

            #
            # Cosmetics.
            #

            ax.grid(
                True,
                alpha=0.25,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            #
            # Bottom row gets x label.
            #

            if row == len(pairings) - 1:
                ax.set_xlabel("Epoch")
            #
            # Show y-axis tick labels for every metric.
            #

            ax.tick_params(
                axis="y",
                labelleft=True,
            )

    #
    # -------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------
    #

    handles, labels = axes[0, 0].get_legend_handles_labels()

    if handles:

        axes[0, 0].legend(
            handles,
            labels,
            frameon=False,
            fontsize=8,
            loc="lower right",
        )

    #
    # -------------------------------------------------------------
    # Figure title
    # -------------------------------------------------------------
    #

    fig.suptitle(
        "Figure 1. Training Curves Across Metrics and Pairings",
        fontsize=14,
    )

    #
    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------
    #

    fig.savefig(
        "figure_1_training_curves.png",
        dpi=300,
        bbox_inches="tight",
    )

    print(
        "Saved Figure 1 to figure_1_training_curves.png"
    )

def _figure_3_best_last_checkpoints(histories):
    """
    Figure 3.

    Layer-wise interpretability evolution between the best and
    final checkpoints.

    Rows
    ----
    (model, dataset) pairings.

    Columns
    -------
    Layer Score
        Best vs Last checkpoint.

    Δ Layer Score
        Best − Last.

    Curves are averaged across runs.

    Shaded regions indicate ±95% confidence intervals.

    Output
    ------
    figure_3_best_last_checkpoints.png
    """

    import math
    from collections import defaultdict

    import numpy as np
    import matplotlib.pyplot as plt

    try:
        from scipy.stats import t

        HAVE_SCIPY = True

    except Exception:

        HAVE_SCIPY = False

    #
    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------
    #

    def mean_ci(values):

        values = np.asarray(
            values,
            dtype=float,
        )

        values = values[
            ~np.isnan(values)
        ]

        if len(values) == 0:
            return np.nan, np.nan

        mean = values.mean()

        if len(values) == 1:
            return mean, np.nan

        std = values.std(ddof=1)
        sem = std / math.sqrt(len(values))

        if HAVE_SCIPY:
            crit = t.ppf(
                0.975,
                len(values) - 1,
            )
        else:
            crit = 1.96

        return mean, crit * sem

    #
    # -------------------------------------------------------------
    # Collect best/last checkpoints
    # -------------------------------------------------------------
    #

    grouped = defaultdict(list)

    for (model, dataset, trainer, run), history in histories.items():

        if not history:
            continue

        last = max(
            history,
            key=lambda s:
                float("-inf")
                if s.epoch is None
                else s.epoch,
        )

        best = max(
            history,
            key=lambda s:
                float("-inf")
                if s.interpretability_score is None
                else s.interpretability_score,
        )

        grouped[
            (model, dataset)
        ].append(
            (
                trainer,
                best,
                last,
            )
        )

    pairings = sorted(grouped.keys())

    if not pairings:
        return

    #
    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------
    #

    fig, axes = plt.subplots(
        len(pairings),
        2,
        figsize=(10, 2.8 * len(pairings)),
        squeeze=False,
        constrained_layout=True,
    )

    trainer_colors = {
        "Standard": "tab:blue",
        "PGD": "tab:red",
    }

    #
    # -------------------------------------------------------------
    # Draw
    # -------------------------------------------------------------
    #

    for row, pairing in enumerate(pairings):

        model, dataset = pairing

        ax_score = axes[row, 0]
        ax_delta = axes[row, 1]

        ax_score.set_ylabel(
            f"{model}\n{dataset}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
            labelpad=28,
        )

        if row == 0:

            ax_score.set_title(
                "Layer Score",
                fontsize=11,
            )

            ax_delta.set_title(
                "Δ Layer Score (Best − Last)",
                fontsize=11,
            )

        #
        # One pair of curves per trainer
        #

        for trainer in sorted(
            {
                t
                for t, _, _
                in grouped[pairing]
            }
        ):

            entries = [

                (best, last)

                for t, best, last

                in grouped[pairing]

                if t == trainer

            ]

            max_layers = max(
                len(best.layers)
                for best, last in entries
            )

            best_mean = []
            best_ci = []

            last_mean = []
            last_ci = []

            delta_mean = []
            delta_ci = []

            for layer_idx in range(max_layers):

                best_values = []
                last_values = []
                delta_values = []

                for best, last in entries:

                    if (
                        layer_idx < len(best.layers)
                        and layer_idx < len(last.layers)
                    ):

                        b = best.layers[layer_idx].score
                        l = last.layers[layer_idx].score

                        best_values.append(b)
                        last_values.append(l)
                        delta_values.append(b - l)

                m, c = mean_ci(best_values)
                best_mean.append(m)
                best_ci.append(c)

                m, c = mean_ci(last_values)
                last_mean.append(m)
                last_ci.append(c)

                m, c = mean_ci(delta_values)
                delta_mean.append(m)
                delta_ci.append(c)

            x = np.arange(max_layers)

            best_mean = np.asarray(best_mean)
            best_ci = np.asarray(best_ci)

            last_mean = np.asarray(last_mean)
            last_ci = np.asarray(last_ci)

            delta_mean = np.asarray(delta_mean)
            delta_ci = np.asarray(delta_ci)

            color = trainer_colors.get(
                trainer,
                None,
            )

            #
            # Best
            #

            ax_score.plot(
                x,
                best_mean,
                color=color,
                linewidth=2,
                label=f"{trainer} Best",
            )

            ax_score.fill_between(
                x,
                best_mean - best_ci,
                best_mean + best_ci,
                color=color,
                alpha=0.20,
            )

            #
            # Last
            #

            ax_score.plot(
                x,
                last_mean,
                "--",
                color=color,
                linewidth=2,
                label=f"{trainer} Last",
            )

            ax_score.fill_between(
                x,
                last_mean - last_ci,
                last_mean + last_ci,
                color=color,
                alpha=0.10,
            )

            #
            # Δ
            #

            ax_delta.plot(
                x,
                delta_mean,
                color=color,
                linewidth=2,
                label=trainer,
            )

            ax_delta.fill_between(
                x,
                delta_mean - delta_ci,
                delta_mean + delta_ci,
                color=color,
                alpha=0.20,
            )

        #
        # Cosmetics
        #

        ax_score.grid(
            True,
            alpha=0.25,
        )

        ax_delta.grid(
            True,
            alpha=0.25,
        )

        ax_delta.axhline(
            0,
            color="black",
            linewidth=1,
            alpha=0.5,
        )

        ax_score.spines["top"].set_visible(False)
        ax_score.spines["right"].set_visible(False)

        ax_delta.spines["top"].set_visible(False)
        ax_delta.spines["right"].set_visible(False)

        ax_score.set_xlabel("Layer")
        ax_delta.set_xlabel("Layer")

        ax_score.set_ylabel("Score")
        ax_delta.set_ylabel("Δ Score")

    #
    # -------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------
    #

    handles, labels = axes[0, 0].get_legend_handles_labels()

    if handles:

        axes[0, 0].legend(
            handles,
            labels,
            frameon=False,
            fontsize=8,
            loc="best",
        )

    #
    # -------------------------------------------------------------
    # Title
    # -------------------------------------------------------------
    #

    fig.suptitle(
        "Figure 3. Layer-wise Interpretability Evolution Between Best and Final Checkpoints",
        fontsize=14,
    )

    #
    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------
    #

    fig.savefig(
        "figure_3_best_last_checkpoints.png",
        dpi=300,
        bbox_inches="tight",
    )

    print(
        "Saved Figure 3 to figure_3_best_last_checkpoints.png"
    )

def _report_significance_test(histories):
    """
    Generate Table 5.

    Paired significance test comparing Standard training against
    adversarial (PGD) training.

    Pairings
    --------
    (model, dataset)

    Metrics
    -------
    Accuracy
    PGD Accuracy
    Interpretability Score
    Interpretability Integral

    Statistical test
    ----------------
    n >= 30 : paired t-test

    otherwise : Wilcoxon signed-rank test

    Outputs
    -------
    significance_table.csv
    significance_table.tex
    """

    import csv
    from collections import defaultdict

    import numpy as np

    try:
        from scipy.stats import (
            ttest_rel,
            wilcoxon,
        )
    except Exception:
        raise RuntimeError(
            "SciPy is required for significance testing."
        )

    # -------------------------------------------------------------
    # Collect final checkpoint from every run
    # -------------------------------------------------------------

    grouped = defaultdict(
        lambda: defaultdict(dict)
    )

    for (
        model,
        dataset,
        trainer,
        run,
    ), history in histories.items():

        if len(history) == 0:
            continue

        latest = max(
            history,
            key=lambda s: (
                float("-inf")
                if s.epoch is None
                else s.epoch
            ),
        )

        grouped[
            (model, dataset)
        ][trainer][run] = latest

    # -------------------------------------------------------------
    # Statistical helper
    # -------------------------------------------------------------

    def paired_test(x, y):

        x = np.asarray(
            x,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        )

        mask = (
            ~np.isnan(x)
            &
            ~np.isnan(y)
        )

        x = x[mask]
        y = y[mask]

        if len(x) < 2:

            return (
                "",
                np.nan,
                np.nan,
                False,
            )

        #
        # Paired t-test
        #

        if len(x) >= 30:

            statistic, p = ttest_rel(
                x,
                y,
                nan_policy="omit",
            )

            return (
                "Paired t-test",
                float(statistic),
                float(p),
                bool(p < 0.05),
            )

        #
        # Wilcoxon signed-rank
        #

        try:

            statistic, p = wilcoxon(
                x,
                y,
                zero_method="wilcox",
            )

        except ValueError:
            #
            # All paired differences are zero.
            #

            statistic = 0.0
            p = 1.0

        return (
            "Wilcoxon",
            float(statistic),
            float(p),
            bool(p < 0.05),
        )

    # -------------------------------------------------------------
    # Build rows
    # -------------------------------------------------------------

    rows = []

    metrics = [
        (
            "Accuracy",
            "accuracy",
        ),
        (
            "PGD Accuracy",
            "pgd_accuracy",
        ),
        (
            "S",
            "interpretability_score",
        ),
        (
            "Integral",
            "interpretability_integral",
        ),
    ]

    for pairing in sorted(grouped.keys()):

        model, dataset = pairing

        standard = grouped[pairing].get(
            "Standard",
            {},
        )

        pgd = grouped[pairing].get(
            "PGD",
            {},
        )

        common_runs = sorted(
            set(standard.keys())
            &
            set(pgd.keys())
        )

        if len(common_runs) < 2:
            continue

        for metric_name, attribute in metrics:

            standard_values = []
            pgd_values = []

            for run in common_runs:

                standard_values.append(
                    getattr(
                        standard[run],
                        attribute,
                        np.nan,
                    )
                )

                pgd_values.append(
                    getattr(
                        pgd[run],
                        attribute,
                        np.nan,
                    )
                )

            (
                test,
                statistic,
                pvalue,
                significant,
            ) = paired_test(
                standard_values,
                pgd_values,
            )

            rows.append(
                dict(
                    pairing=f"{model}/{dataset}",
                    metric=metric_name,
                    test=test,
                    statistic=statistic,
                    pvalue=pvalue,
                    significant="Yes"
                    if significant
                    else "No",
                )
            )

    # -------------------------------------------------------------
    # Console
    # -------------------------------------------------------------

    headers = [
        "Pairing",
        "Metric",
        "Test",
        "Statistic",
        "p-value",
        "Significant",
    ]

    widths = [
        36,
        18,
        18,
        16,
        14,
        14,
    ]

    line = "".join(
        h.ljust(w)
        for h, w in zip(
            headers,
            widths,
        )
    )

    print()
    print("=" * len(line))
    print(line)
    print("=" * len(line))

    for r in rows:

        print(
            "".join(
                str(v).ljust(w)
                for v, w in zip(
                    [
                        r["pairing"],
                        r["metric"],
                        r["test"],
                        f"{r['statistic']:.4f}",
                        f"{r['pvalue']:.4g}",
                        r["significant"],
                    ],
                    widths,
                )
            )
        )

    print("=" * len(line))
    print()

    # -------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------

    with open(
        "significance_table.csv",
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "Pairing",
                "Metric",
                "Test",
                "Test statistic",
                "p-value",
                "Significant (alpha=0.05)",
            ]
        )

        for r in rows:

            writer.writerow(
                [
                    r["pairing"],
                    r["metric"],
                    r["test"],
                    r["statistic"],
                    r["pvalue"],
                    r["significant"],
                ]
            )

    # -------------------------------------------------------------
    # LaTeX
    # -------------------------------------------------------------

    with open(
        "significance_table.tex",
        "w",
    ) as f:

        f.write("\\begin{tabular}{lllccc}\n")
        f.write("\\hline\n")
        f.write(
            "Pairing & "
            "Metric & "
            "Test & "
            "Test statistic & "
            "$p$-value & "
            "Significant ($\\alpha=0.05$)"
            "\\\\\n"
        )
        f.write("\\hline\n")

        previous_pairing = None

        for r in rows:

            pairing = (
                r["pairing"]
                if r["pairing"] != previous_pairing
                else ""
            )

            previous_pairing = r["pairing"]

            f.write(
                f"{pairing} & "
                f"{r['metric']} & "
                f"{r['test']} & "
                f"{r['statistic']:.4f} & "
                f"{r['pvalue']:.4g} & "
                f"{r['significant']}"
                "\\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    print("Saved significance_table.csv")
    print("Saved significance_table.tex")

def _figure_2_stability(histories):
    """
    Figure 2.

    Stability of interpretability score S across replicate runs.

    Layout
    ------
    Columns:
        (model, dataset) pairings

    Rows:
        Standard training
        Adversarial / PGD training

    Curves
    ------
    Thin lines:
        Individual replicate runs.

    Thick black line:
        Mean across replicate runs.

    Metric
    ------
    S = interpretability_score

    Output
    ------
    figure_2_stability.png
    """

    from collections import defaultdict

    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------
    # Organize snapshots
    #
    # pairing -> trainer -> run -> epoch -> S
    # -------------------------------------------------------------

    grouped = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(dict)
        )
    )

    for (model, dataset, trainer, run), history in histories.items():

        pairing = (model, dataset)

        for snapshot in history:

            if snapshot.epoch is None:
                continue

            score = snapshot.interpretability_score

            if score is None:
                continue

            grouped[pairing][trainer][run][snapshot.epoch] = float(score)

    # -------------------------------------------------------------
    # Determine pairings
    # -------------------------------------------------------------

    pairings = sorted(grouped.keys())

    if not pairings:
        print("No data available for Figure 2.")
        return

    # -------------------------------------------------------------
    # Figure
    #
    # Two rows:
    #   0 = Standard
    #   1 = PGD / adversarial
    #
    # One column per pairing.
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        len(pairings),
        figsize=(2.15 * len(pairings), 4.2),
        squeeze=False,
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )

    # -------------------------------------------------------------
    # Trainer aliases
    # -------------------------------------------------------------

    standard_names = {
        "Standard",
        "standard",
    }

    adversarial_names = {
        "PGD",
        "pgd",
        "Adversarial",
        "adversarial",
    }

    def trainer_type(trainer):

        name = str(trainer)

        if name in standard_names:
            return "Standard"

        if name in adversarial_names:
            return "PGD"

        #
        # Preserve compatibility with other trainer names.
        #

        lowered = name.lower()

        if "pgd" in lowered or "adv" in lowered:
            return "PGD"

        if "standard" in lowered:
            return "Standard"

        return name

    # -------------------------------------------------------------
    # Draw one panel
    # -------------------------------------------------------------

    for col, pairing in enumerate(pairings):

        model, dataset = pairing

        trainer_groups = {
            "Standard": [],
            "PGD": [],
        }

        #
        # Collect all runs for this pairing.
        #

        for trainer, runs in grouped[pairing].items():

            kind = trainer_type(trainer)

            if kind not in trainer_groups:
                continue

            for run, epoch_values in runs.items():
                trainer_groups[kind].append(
                    (run, epoch_values)
                )

        # ---------------------------------------------------------
        # Panel labels
        # ---------------------------------------------------------

        axes[0, col].set_title(
            f"{model} {dataset}",
            fontsize=8,
        )

        # ---------------------------------------------------------
        # Standard / PGD rows
        # ---------------------------------------------------------

        for row, kind in enumerate(
            ["Standard", "PGD"]
        ):

            ax = axes[row, col]

            runs = trainer_groups[kind]

            #
            # Draw individual replicate runs.
            #

            for run, epoch_values in sorted(
                runs,
                key=lambda item: str(item[0])
            ):

                epochs = sorted(epoch_values.keys())

                values = [
                    epoch_values[epoch]
                    for epoch in epochs
                ]

                ax.plot(
                    epochs,
                    values,
                    linewidth=0.8,
                    alpha=0.45,
                    color="black",
                )

            # -----------------------------------------------------
            # Compute mean at each epoch.
            # -----------------------------------------------------

            if runs:

                epoch_values = defaultdict(list)

                for run, values in runs:

                    for epoch, score in values.items():
                        epoch_values[epoch].append(score)

                epochs = sorted(epoch_values.keys())

                mean = np.asarray(
                    [
                        np.mean(epoch_values[epoch])
                        for epoch in epochs
                    ],
                    dtype=float,
                )

                #
                # Thick mean curve.
                #

                ax.plot(
                    epochs,
                    mean,
                    color="black",
                    linewidth=2.2,
                    zorder=10,
                )

            # -----------------------------------------------------
            # Cosmetics
            # -----------------------------------------------------

            ax.set_ylim(0.33, 0.82)

            ax.grid(
                False,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.tick_params(
                labelsize=7,
            )

            #
            # Match Figure 2: epoch labels on x axis.
            #

            ax.set_xlim(
                left=0,
            )

            if row == 1:
                ax.set_xlabel(
                    "Epoch",
                    fontsize=8,
                )

            #
            # Row labels only on the first column.
            #

            if col == 0:

                ax.set_ylabel(
                    kind,
                    fontsize=8,
                )

    # -------------------------------------------------------------
    # Shared y-axis label
    # -------------------------------------------------------------

    fig.text(
        0.012,
        0.50,
        "S",
        rotation="vertical",
        va="center",
        fontsize=10,
    )

    # -------------------------------------------------------------
    # Figure title
    # -------------------------------------------------------------

    fig.suptitle(
        "Figure 2. Stability of S across replicate runs",
        fontsize=11,
    )

    # -------------------------------------------------------------
    # Caption
    # -------------------------------------------------------------

    fig.text(
        0.5,
        -0.025,
        "Thin lines: individual runs; thick line: mean. "
        "Standard (top) vs. adversarial (bottom).",
        ha="center",
        fontsize=8,
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    fig.savefig(
        "figure_2_stability.png",
        dpi=300,
        bbox_inches="tight",
    )

    print(
        "Saved Figure 2 to figure_2_stability.png"
    )


def _figure_III_trade_offs(
    histories,
    checkpoint="best",
):
    """
    Figure 3.

    Accuracy × robustness × interpretability trade-offs.

    Parameters
    ----------
    histories : dict
        Output of Report.histories().

    checkpoint : {"best", "last", "all"}, default="best"
        Which checkpoints to use.

        "best":
            One checkpoint per run: the checkpoint with maximum S,
            where S = interpretability_score.

        "last":
            One checkpoint per run: the checkpoint with the
            largest epoch.

        "all":
            Every valid checkpoint from every run.

    Panels
    ------
    (a) Accuracy vs. S
    (b) PGD Accuracy vs. S
    (c) Accuracy vs. PGD Accuracy
    (d) 3-way scatter:
            x = Accuracy
            y = PGD Accuracy
            color = S

    For "best" and "last"
    ---------------------
    Individual runs are shown as points.

    A mean marker (×) is shown for each
    (model, dataset, trainer) group.

    For "all"
    ---------
    All epoch/run observations are shown as small,
    semi-transparent points.

    Mean trajectories are calculated epoch-by-epoch
    within each (model, dataset, trainer) group.

    Output
    ------
    figure_3_best_trade_offs.png
    figure_3_last_trade_offs.png
    figure_3_all_trade_offs.png
    """

    from collections import defaultdict

    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------
    # Validate mode
    # -------------------------------------------------------------

    if checkpoint not in {
        "best",
        "last",
        "all",
    }:
        raise ValueError(
            "checkpoint must be one of "
            "'best', 'last', or 'all'."
        )

    # -------------------------------------------------------------
    # Normalize trainer names
    # -------------------------------------------------------------

    def trainer_type(trainer):
        """
        Normalize trainer names to Figure 3 categories.
        """

        name = str(trainer)
        lowered = name.lower()

        if (
            name in {"Standard", "standard"}
            or "standard" in lowered
        ):
            return "Standard"

        if (
            name in {
                "PGD",
                "pgd",
                "Adversarial",
                "adversarial",
            }
            or "pgd" in lowered
            or "adv" in lowered
        ):
            return "Adversarial"

        return name

    # -------------------------------------------------------------
    # Select best/last checkpoint
    # -------------------------------------------------------------

    def select_checkpoint(history):

        if not history:
            return None

        if checkpoint == "last":

            return max(
                history,
                key=lambda snapshot: (
                    float("-inf")
                    if snapshot.epoch is None
                    else snapshot.epoch
                ),
            )

        if checkpoint == "best":

            valid = [
                snapshot
                for snapshot in history
                if snapshot.interpretability_score
                is not None
            ]

            if not valid:
                return None

            return max(
                valid,
                key=lambda snapshot: float(
                    snapshot.interpretability_score
                ),
            )

        raise RuntimeError(
            "select_checkpoint() should not be called "
            "for checkpoint='all'."
        )

    # -------------------------------------------------------------
    # Convert snapshot to a plotting point
    # -------------------------------------------------------------

    def snapshot_to_point(
        snapshot,
        model,
        dataset,
        trainer,
        run,
    ):

        accuracy = snapshot.accuracy
        pgd_accuracy = snapshot.pgd_accuracy
        score = snapshot.interpretability_score

        if (
            accuracy is None
            or pgd_accuracy is None
            or score is None
        ):
            return None

        try:

            accuracy = float(accuracy)
            pgd_accuracy = float(pgd_accuracy)
            score = float(score)

        except (TypeError, ValueError):

            return None

        if not all(
            np.isfinite(value)
            for value in (
                accuracy,
                pgd_accuracy,
                score,
            )
        ):
            return None

        return {
            "model": model,
            "dataset": dataset,
            "trainer": trainer,
            "run": run,
            "epoch": snapshot.epoch,
            "accuracy": accuracy,
            "pgd_accuracy": pgd_accuracy,
            "score": score,
        }

    # -------------------------------------------------------------
    # Collect observations
    # -------------------------------------------------------------

    points = []

    for (
        model,
        dataset,
        trainer,
        run,
    ), history in histories.items():

        normalized_trainer = trainer_type(trainer)

        # ---------------------------------------------------------
        # ALL
        # ---------------------------------------------------------

        if checkpoint == "all":

            for snapshot in history:

                point = snapshot_to_point(
                    snapshot,
                    model,
                    dataset,
                    normalized_trainer,
                    run,
                )

                if point is not None:
                    points.append(point)

        # ---------------------------------------------------------
        # BEST / LAST
        # ---------------------------------------------------------

        else:

            snapshot = select_checkpoint(history)

            if snapshot is None:
                continue

            point = snapshot_to_point(
                snapshot,
                model,
                dataset,
                normalized_trainer,
                run,
            )

            if point is not None:
                points.append(point)

    if not points:

        print(
            "No complete Accuracy / PGD Accuracy / S "
            f"data available for Figure 3 "
            f"({checkpoint})."
        )

        return

    # -------------------------------------------------------------
    # Colors
    # -------------------------------------------------------------

    trainer_colors = {
        "Standard": "tab:blue",
        "Adversarial": "tab:red",
    }

    # -------------------------------------------------------------
    # Group observations
    #
    # For best/last:
    #     group -> runs
    #
    # For all:
    #     group -> epochs -> observations
    # -------------------------------------------------------------

    grouped = defaultdict(list)

    for point in points:

        key = (
            point["model"],
            point["dataset"],
            point["trainer"],
        )

        grouped[key].append(point)

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(12, 3.2),
        constrained_layout=True,
    )

    ax_a, ax_b, ax_c, ax_d = axes

    # -------------------------------------------------------------
    # Individual scatter helper
    # -------------------------------------------------------------

    def plot_points(
        ax,
        group,
        x_key,
        y_key,
        trainer,
    ):

        if not group:
            return

        ax.scatter(
            [
                point[x_key]
                for point in group
            ],
            [
                point[y_key]
                for point in group
            ],
            s=28 if checkpoint != "all" else 16,
            alpha=(
                0.85
                if checkpoint != "all"
                else 0.25
            ),
            color=trainer_colors.get(
                trainer,
                "black",
            ),
            edgecolors="none",
        )

    # -------------------------------------------------------------
    # Mean marker for BEST / LAST
    # -------------------------------------------------------------

    def plot_mean_marker(
        ax,
        group,
        x_key,
        y_key,
        trainer,
    ):

        if not group:
            return

        x = np.asarray(
            [
                point[x_key]
                for point in group
            ],
            dtype=float,
        )

        y = np.asarray(
            [
                point[y_key]
                for point in group
            ],
            dtype=float,
        )

        ax.scatter(
            np.mean(x),
            np.mean(y),
            marker="x",
            s=70,
            linewidths=2.0,
            color=trainer_colors.get(
                trainer,
                "black",
            ),
            zorder=20,
        )

    # -------------------------------------------------------------
    # Mean trajectory for ALL
    #
    # At each epoch:
    #
    #     mean Accuracy
    #     mean PGD Accuracy
    #     mean S
    #
    # across replicate runs in the group.
    # -------------------------------------------------------------

    def epoch_means(group):

        by_epoch = defaultdict(list)

        for point in group:

            if point["epoch"] is None:
                continue

            by_epoch[point["epoch"]].append(point)

        means = []

        for epoch in sorted(by_epoch):

            epoch_points = by_epoch[epoch]

            means.append(
                {
                    "epoch": epoch,

                    "accuracy": np.mean(
                        [
                            p["accuracy"]
                            for p in epoch_points
                        ]
                    ),

                    "pgd_accuracy": np.mean(
                        [
                            p["pgd_accuracy"]
                            for p in epoch_points
                        ]
                    ),

                    "score": np.mean(
                        [
                            p["score"]
                            for p in epoch_points
                        ]
                    ),
                }
            )

        return means

    # -------------------------------------------------------------
    # Plot panels (a)-(c)
    # -------------------------------------------------------------

    for (
        model,
        dataset,
        trainer,
    ), group in sorted(grouped.items()):

        # ---------------------------------------------------------
        # Individual observations
        # ---------------------------------------------------------

        plot_points(
            ax_a,
            group,
            "accuracy",
            "score",
            trainer,
        )

        plot_points(
            ax_b,
            group,
            "pgd_accuracy",
            "score",
            trainer,
        )

        plot_points(
            ax_c,
            group,
            "accuracy",
            "pgd_accuracy",
            trainer,
        )

        # ---------------------------------------------------------
        # Best / Last mean
        # ---------------------------------------------------------

        if checkpoint in {
            "best",
            "last",
        }:

            plot_mean_marker(
                ax_a,
                group,
                "accuracy",
                "score",
                trainer,
            )

            plot_mean_marker(
                ax_b,
                group,
                "pgd_accuracy",
                "score",
                trainer,
            )

            plot_mean_marker(
                ax_c,
                group,
                "accuracy",
                "pgd_accuracy",
                trainer,
            )

        # ---------------------------------------------------------
        # ALL mean trajectory
        # ---------------------------------------------------------

        else:

            means = epoch_means(group)

            if not means:
                continue

            mean_accuracy = [
                point["accuracy"]
                for point in means
            ]

            mean_pgd = [
                point["pgd_accuracy"]
                for point in means
            ]

            mean_score = [
                point["score"]
                for point in means
            ]

            color = trainer_colors.get(
                trainer,
                "black",
            )

            #
            # Accuracy vs S
            #

            ax_a.plot(
                mean_accuracy,
                mean_score,
                color=color,
                linewidth=2.0,
                alpha=0.95,
            )

            #
            # PGD Accuracy vs S
            #

            ax_b.plot(
                mean_pgd,
                mean_score,
                color=color,
                linewidth=2.0,
                alpha=0.95,
            )

            #
            # Accuracy vs PGD Accuracy
            #

            ax_c.plot(
                mean_accuracy,
                mean_pgd,
                color=color,
                linewidth=2.0,
                alpha=0.95,
            )

    # -------------------------------------------------------------
    # Panel D
    #
    # All individual observations are colored by S.
    # This is deliberately independent of trainer color.
    # -------------------------------------------------------------

    all_accuracy = np.asarray(
        [
            point["accuracy"]
            for point in points
        ],
        dtype=float,
    )

    all_pgd = np.asarray(
        [
            point["pgd_accuracy"]
            for point in points
        ],
        dtype=float,
    )

    all_score = np.asarray(
        [
            point["score"]
            for point in points
        ],
        dtype=float,
    )

    scatter = ax_d.scatter(
        all_accuracy,
        all_pgd,
        c=all_score,
        cmap="viridis",
        s=30 if checkpoint != "all" else 18,
        alpha=0.9 if checkpoint != "all" else 0.35,
        edgecolors="none",
    )

    # -------------------------------------------------------------
    # Mean trajectory in panel D for ALL
    # -------------------------------------------------------------

    if checkpoint == "all":

        for (
            model,
            dataset,
            trainer,
        ), group in sorted(grouped.items()):

            means = epoch_means(group)

            if not means:
                continue

            ax_d.plot(
                [
                    point["accuracy"]
                    for point in means
                ],
                [
                    point["pgd_accuracy"]
                    for point in means
                ],
                color=trainer_colors.get(
                    trainer,
                    "black",
                ),
                linewidth=2.0,
                alpha=0.9,
            )

    colorbar = fig.colorbar(
        scatter,
        ax=ax_d,
        pad=0.02,
    )

    colorbar.set_label("S")

    # -------------------------------------------------------------
    # Axis limits
    # -------------------------------------------------------------

    def padded_limits(values):

        values = np.asarray(
            values,
            dtype=float,
        )

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:
            return (0.0, 1.0)

        low = np.min(values)
        high = np.max(values)

        if low == high:

            margin = 0.05 * max(
                abs(low),
                1.0,
            )

        else:

            margin = 0.08 * (
                high - low
            )

        return (
            low - margin,
            high + margin,
        )

    accuracy_limits = padded_limits(
        all_accuracy
    )

    pgd_limits = padded_limits(
        all_pgd
    )

    score_limits = padded_limits(
        all_score
    )

    ax_a.set_xlim(accuracy_limits)
    ax_a.set_ylim(score_limits)

    ax_b.set_xlim(pgd_limits)
    ax_b.set_ylim(score_limits)

    ax_c.set_xlim(accuracy_limits)
    ax_c.set_ylim(pgd_limits)

    ax_d.set_xlim(accuracy_limits)
    ax_d.set_ylim(pgd_limits)

    # -------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------

    ax_a.set_xlabel("Accuracy")
    ax_a.set_ylabel("S")
    ax_a.set_title(
        "(a) Accuracy vs. S",
        fontsize=9,
    )

    ax_b.set_xlabel("PGD Accuracy")
    ax_b.set_ylabel("S")
    ax_b.set_title(
        "(b) PGD Accuracy vs. S",
        fontsize=9,
    )

    ax_c.set_xlabel("Accuracy")
    ax_c.set_ylabel("PGD Accuracy")
    ax_c.set_title(
        "(c) Accuracy vs. PGD Accuracy",
        fontsize=9,
    )

    ax_d.set_xlabel("Accuracy")
    ax_d.set_ylabel("PGD Accuracy")
    ax_d.set_title(
        "(d) 3-way scatter",
        fontsize=9,
    )

    # -------------------------------------------------------------
    # Cosmetics
    # -------------------------------------------------------------

    for ax in axes:

        ax.grid(
            True,
            alpha=0.20,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(
            labelsize=8,
        )

    # -------------------------------------------------------------
    # Legend
    # -------------------------------------------------------------

    handles = []

    for trainer in (
        "Standard",
        "Adversarial",
    ):

        if any(
            point["trainer"] == trainer
            for point in points
        ):

            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=trainer_colors[trainer],
                    markersize=5,
                    label=trainer,
                )
            )

    if checkpoint in {
        "best",
        "last",
    }:

        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="x",
                linestyle="",
                color="black",
                markersize=7,
                markeredgewidth=2,
                label="Mean",
            )
        )

    else:

        handles.append(
            plt.Line2D(
                [0],
                [0],
                linestyle="-",
                color="black",
                linewidth=2,
                label="Epoch-wise mean",
            )
        )

    if handles:

        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=len(handles),
            frameon=False,
            fontsize=8,
        )

    # -------------------------------------------------------------
    # Title
    # -------------------------------------------------------------

    checkpoint_label = {
        "best": "Best-S checkpoint",
        "last": "Last checkpoint",
        "all": "All checkpoints",
    }[checkpoint]

    fig.suptitle(
        "Figure 3. Accuracy × robustness × "
        f"interpretability trade-offs "
        f"({checkpoint_label})",
        fontsize=10,
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    filename = {
        "best": "figure_3_best_trade_offs.png",
        "last": "figure_3_last_trade_offs.png",
        "all": "figure_3_all_trade_offs.png",
    }[checkpoint]

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved Figure 3 to {filename}"
    )

    plt.close(fig)

def _figure_4_distros(
    histories,
    checkpoint="best",
):
    """
    Figure 4.

    Per-pairing distributions with raw observations overlaid
    on violin plots.

    Parameters
    ----------
    histories : dict
        Output of Report.histories().

    checkpoint : {"best", "last", "all"}, default="best"
        Checkpoint selection.

        "best":
            One observation per run at maximum S.

        "last":
            One observation per run at the final epoch.

        "all":
            Every valid snapshot is included.

    Panels
    ------
    (a) Accuracy
    (b) PGD Accuracy
    (c) S (Interpretability)
    (d) Integral

    At each model/dataset pairing:
        Standard and Adversarial distributions are overlaid.

    The violin shows the distribution and the raw observations
    are jittered horizontally over the violin.

    Output
    ------
    figure_4_distros.png
    """

    from collections import defaultdict

    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------
    # Validate checkpoint mode
    # -------------------------------------------------------------

    if checkpoint not in {
        "best",
        "last",
        "all",
    }:
        raise ValueError(
            "checkpoint must be one of "
            "'best', 'last', or 'all'."
        )

    # -------------------------------------------------------------
    # Normalize trainer names
    # -------------------------------------------------------------

    def trainer_type(trainer):

        name = str(trainer)
        lowered = name.lower()

        if (
            name in {"Standard", "standard"}
            or "standard" in lowered
        ):
            return "Standard"

        if (
            name in {
                "PGD",
                "pgd",
                "Adversarial",
                "adversarial",
            }
            or "pgd" in lowered
            or "adv" in lowered
        ):
            return "Adversarial"

        return name

    # -------------------------------------------------------------
    # Select checkpoint
    # -------------------------------------------------------------

    def select_checkpoint(history):

        if not history:
            return None

        if checkpoint == "last":

            return max(
                history,
                key=lambda snapshot: (
                    float("-inf")
                    if snapshot.epoch is None
                    else snapshot.epoch
                ),
            )

        if checkpoint == "best":

            valid = [
                snapshot
                for snapshot in history
                if snapshot.interpretability_score
                is not None
            ]

            if not valid:
                return None

            return max(
                valid,
                key=lambda snapshot: float(
                    snapshot.interpretability_score
                ),
            )

        return None

    # -------------------------------------------------------------
    # Extract one observation
    # -------------------------------------------------------------

    def make_observation(
        snapshot,
        model,
        dataset,
        trainer,
        run,
    ):

        values = {
            "accuracy": snapshot.accuracy,
            "pgd_accuracy": snapshot.pgd_accuracy,
            "score": snapshot.interpretability_score,
            "integral": snapshot.interpretability_integral,
        }

        #
        # Convert all values to float.
        #

        for key, value in values.items():

            if value is None:
                return None

            try:
                value = float(value)
            except (TypeError, ValueError):
                return None

            if not np.isfinite(value):
                return None

            values[key] = value

        return {
            "model": model,
            "dataset": dataset,
            "trainer": trainer,
            "run": run,
            "epoch": snapshot.epoch,
            **values,
        }

    # -------------------------------------------------------------
    # Collect observations
    # -------------------------------------------------------------

    observations = []

    for (
        model,
        dataset,
        trainer,
        run,
    ), history in histories.items():

        normalized_trainer = trainer_type(trainer)

        # ---------------------------------------------------------
        # ALL
        # ---------------------------------------------------------

        if checkpoint == "all":

            for snapshot in history:

                observation = make_observation(
                    snapshot,
                    model,
                    dataset,
                    normalized_trainer,
                    run,
                )

                if observation is not None:
                    observations.append(
                        observation
                    )

        # ---------------------------------------------------------
        # BEST / LAST
        # ---------------------------------------------------------

        else:

            snapshot = select_checkpoint(
                history
            )

            if snapshot is None:
                continue

            observation = make_observation(
                snapshot,
                model,
                dataset,
                normalized_trainer,
                run,
            )

            if observation is not None:
                observations.append(
                    observation
                )

    if not observations:

        print(
            "No complete Accuracy / PGD Accuracy / "
            "S / Integral data available for Figure 4 "
            f"({checkpoint})."
        )

        return

    # -------------------------------------------------------------
    # Pairings
    # -------------------------------------------------------------

    pairings = sorted(
        set(
            (
                observation["model"],
                observation["dataset"],
            )
            for observation in observations
        )
    )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------

    metrics = [
        (
            "accuracy",
            "Accuracy",
        ),
        (
            "pgd_accuracy",
            "PGD Accuracy",
        ),
        (
            "score",
            "S (Interpretability)",
        ),
        (
            "integral",
            "Integral",
        ),
    ]

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(
            max(11.0, 1.45 * len(pairings)),
            3.8,
        ),
        sharex=True,
        constrained_layout=True,
    )

    # -------------------------------------------------------------
    # Trainer appearance
    # -------------------------------------------------------------

    trainer_colors = {
        "Standard": "tab:blue",
        "Adversarial": "tab:red",
    }

    # Horizontal offset for the two distributions.
    #
    # Standard:
    #     left
    #
    # Adversarial:
    #     right
    # -------------------------------------------------------------

    offset = {
        "Standard": -0.18,
        "Adversarial": 0.18,
    }

    width = 0.32

    # -------------------------------------------------------------
    # Plot each metric
    # -------------------------------------------------------------

    for ax, (metric, ylabel) in zip(
        axes,
        metrics,
    ):

        # ---------------------------------------------------------
        # Violin data
        # ---------------------------------------------------------

        violin_data = []
        violin_positions = []
        violin_colors = []

        # ---------------------------------------------------------
        # Raw observations
        # ---------------------------------------------------------

        for pairing_index, pairing in enumerate(
            pairings
        ):

            model, dataset = pairing

            for trainer in (
                "Standard",
                "Adversarial",
            ):

                values = [
                    observation[metric]
                    for observation in observations
                    if (
                        observation["model"],
                        observation["dataset"],
                    ) == pairing
                    and observation["trainer"] == trainer
                ]

                if not values:
                    continue

                position = (
                    pairing_index
                    + offset[trainer]
                )

                violin_data.append(
                    values
                )

                violin_positions.append(
                    position
                )

                violin_colors.append(
                    trainer_colors[trainer]
                )

        # ---------------------------------------------------------
        # Violin plots
        # ---------------------------------------------------------

        if violin_data:

            violins = ax.violinplot(
                violin_data,
                positions=violin_positions,
                widths=width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )

            #
            # Color each violin according to trainer.
            #

            for body, color in zip(
                violins["bodies"],
                violin_colors,
            ):

                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.35)

        # ---------------------------------------------------------
        # Raw observations
        # ---------------------------------------------------------

        rng = np.random.default_rng(42)

        for pairing_index, pairing in enumerate(
            pairings
        ):

            for trainer in (
                "Standard",
                "Adversarial",
            ):

                values = [
                    observation[metric]
                    for observation in observations
                    if (
                        observation["model"],
                        observation["dataset"],
                    ) == pairing
                    and observation["trainer"] == trainer
                ]

                if not values:
                    continue

                #
                # Deterministic horizontal jitter.
                #

                jitter = rng.normal(
                    0.0,
                    0.035,
                    size=len(values),
                )

                x = (
                    pairing_index
                    + offset[trainer]
                    + jitter
                )

                ax.scatter(
                    x,
                    values,
                    s=9,
                    alpha=0.65,
                    color=trainer_colors[trainer],
                    edgecolors="none",
                    zorder=5,
                )

        # ---------------------------------------------------------
        # Axes
        # ---------------------------------------------------------

        ax.set_title(
            ylabel,
            fontsize=9,
        )

        ax.set_ylabel(
            ylabel,
            fontsize=8,
        )

        ax.grid(
            axis="y",
            alpha=0.20,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(
            labelsize=7,
        )

        # ---------------------------------------------------------
        # X-axis
        # ---------------------------------------------------------

        ax.set_xticks(
            range(len(pairings))
        )

        ax.set_xticklabels(
            [
                f"{model}-{dataset}"
                for model, dataset in pairings
            ],
            rotation=60,
            ha="right",
            fontsize=6,
        )

        ax.set_xlim(
            -0.6,
            len(pairings) - 0.4,
        )

    # -------------------------------------------------------------
    # Shared legend
    # -------------------------------------------------------------

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=trainer_colors["Standard"],
            markersize=5,
            label="Standard",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=trainer_colors["Adversarial"],
            markersize=5,
            label="Adversarial",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    # -------------------------------------------------------------
    # Title
    # -------------------------------------------------------------

    checkpoint_label = {
        "best": "Best-S checkpoint",
        "last": "Last checkpoint",
        "all": "All checkpoints",
    }[checkpoint]

    fig.suptitle(
        "Figure 4. Per-pairing distributions "
        f"({checkpoint_label})",
        fontsize=10,
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    filename = {
        "best": "figure_4_distros.png",
        "last": "figure_4_distros_last.png",
        "all": "figure_4_distros_all.png",
    }[checkpoint]

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved Figure 4 to {filename}"
    )

    plt.close(fig)

def _figure_5_layerwise_score(
    histories,
    checkpoint="best",
):
    """
    Figure 5.

    Layer-wise interpretability score S, comparing
    Standard and Adversarial training.

    Parameters
    ----------
    histories : dict
        Output of Report.histories().

    checkpoint : {"best", "last", "all"}, default="best"
        Checkpoint selection.

        "best":
            Select the checkpoint with maximum
            snapshot.interpretability_score for each run,
            then plot the LayerReport.score values.

        "last":
            Select the final checkpoint for each run,
            then plot the LayerReport.score values.

        "all":
            Plot layer-wise scores from every checkpoint.

    Figure
    ------
    One panel per (model, dataset) pairing.

    Thin lines:
        Individual runs.

    Thick line:
        Mean across runs.

    Standard:
        Blue.

    Adversarial:
        Red.

    Output
    ------
    figure_5_layerwise_score.png
        for checkpoint="best"

    figure_5_layerwise_score_last.png
        for checkpoint="last"

    figure_5_layerwise_score_all.png
        for checkpoint="all"
    """

    from collections import defaultdict

    import numpy as np
    import matplotlib.pyplot as plt

    # ---------------------------------------------------------
    # Validate checkpoint
    # ---------------------------------------------------------

    if checkpoint not in {
        "best",
        "last",
        "all",
    }:
        raise ValueError(
            "checkpoint must be one of "
            "'best', 'last', or 'all'."
        )

    # ---------------------------------------------------------
    # Normalize trainer names
    # ---------------------------------------------------------

    def trainer_type(trainer):

        name = str(trainer)
        lowered = name.lower()

        if (
            name in {"Standard", "standard"}
            or "standard" in lowered
        ):
            return "Standard"

        if (
            name in {
                "PGD",
                "pgd",
                "Adversarial",
                "adversarial",
            }
            or "pgd" in lowered
            or "adv" in lowered
        ):
            return "Adversarial"

        return name

    # ---------------------------------------------------------
    # Checkpoint selection
    # ---------------------------------------------------------

    def select_checkpoint(history):

        if not history:
            return None

        if checkpoint == "last":

            return max(
                history,
                key=lambda snapshot: (
                    float("-inf")
                    if snapshot.epoch is None
                    else snapshot.epoch
                ),
            )

        if checkpoint == "best":

            valid = [
                snapshot
                for snapshot in history
                if snapshot.interpretability_score
                is not None
            ]

            if not valid:
                return None

            return max(
                valid,
                key=lambda snapshot: float(
                    snapshot.interpretability_score
                ),
            )

        return None

    # ---------------------------------------------------------
    # Extract layer-wise scores
    # ---------------------------------------------------------

    def extract_layer_scores(snapshot):

        layer_scores = []

        for layer_report in snapshot.layers:

            if layer_report.score is None:
                continue

            try:
                score = float(
                    layer_report.score
                )
            except (TypeError, ValueError):
                continue

            if not np.isfinite(score):
                continue

            layer_scores.append(
                (
                    str(layer_report.layer),
                    score,
                )
            )

        return layer_scores

    # ---------------------------------------------------------
    # Collect observations
    #
    # Each observation corresponds to:
    #
    #   run × epoch × layer
    #
    # for "all", and
    #
    #   run × layer
    #
    # for "best"/"last".
    # ---------------------------------------------------------

    observations = []

    for (
        model,
        dataset,
        trainer,
        run,
    ), history in histories.items():

        trainer = trainer_type(trainer)

        # -----------------------------------------------------
        # ALL checkpoints
        # -----------------------------------------------------

        if checkpoint == "all":

            selected_snapshots = history

        # -----------------------------------------------------
        # BEST / LAST
        # -----------------------------------------------------

        else:

            snapshot = select_checkpoint(
                history
            )

            if snapshot is None:
                continue

            selected_snapshots = [
                snapshot
            ]

        # -----------------------------------------------------
        # Extract layer scores
        # -----------------------------------------------------

        for snapshot in selected_snapshots:

            layer_scores = extract_layer_scores(
                snapshot
            )

            for layer_index, (
                layer_name,
                score,
            ) in enumerate(layer_scores):

                observations.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "trainer": trainer,
                        "run": run,
                        "epoch": snapshot.epoch,
                        "layer": layer_name,
                        "layer_index": layer_index,
                        "score": score,
                    }
                )

    if not observations:

        print(
            "No layer-wise score data available "
            f"for Figure 5 ({checkpoint})."
        )

        return

    # ---------------------------------------------------------
    # Pairings
    # ---------------------------------------------------------

    pairings = sorted(
        set(
            (
                observation["model"],
                observation["dataset"],
            )
            for observation in observations
        )
    )

    # ---------------------------------------------------------
    # Figure layout
    # ---------------------------------------------------------

    n_panels = len(pairings)

    ncols = min(
        4,
        n_panels,
    )

    nrows = int(
        np.ceil(
            n_panels / ncols
        )
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            3.1 * ncols,
            2.6 * nrows,
        ),
        squeeze=False,
        constrained_layout=True,
    )

    axes = axes.ravel()

    # ---------------------------------------------------------
    # Colors
    # ---------------------------------------------------------

    trainer_colors = {
        "Standard": "tab:blue",
        "Adversarial": "tab:red",
    }

    # ---------------------------------------------------------
    # Plot each model/dataset pairing
    # ---------------------------------------------------------

    for ax, pairing in zip(
        axes,
        pairings,
    ):

        model, dataset = pairing

        pairing_observations = [
            observation
            for observation in observations
            if (
                observation["model"],
                observation["dataset"],
            ) == pairing
        ]

        # -----------------------------------------------------
        # Determine layer ordering
        # -----------------------------------------------------

        layer_info = {}

        for observation in pairing_observations:

            layer_info[
                observation["layer"]
            ] = observation["layer_index"]

        layers = [
            layer
            for layer, _ in sorted(
                layer_info.items(),
                key=lambda item: item[1],
            )
        ]

        layer_to_x = {
            layer: index
            for index, layer in enumerate(
                layers
            )
        }

        # -----------------------------------------------------
        # Individual runs
        # -----------------------------------------------------

        for trainer in (
            "Standard",
            "Adversarial",
        ):

            trainer_observations = [
                observation
                for observation
                in pairing_observations
                if observation["trainer"] == trainer
            ]

            if not trainer_observations:
                continue

            color = trainer_colors.get(
                trainer,
                "black",
            )

            # -------------------------------------------------
            # Group by run
            # -------------------------------------------------

            runs = defaultdict(list)

            for observation in trainer_observations:

                runs[
                    observation["run"]
                ].append(observation)

            # -------------------------------------------------
            # Draw individual run profiles
            # -------------------------------------------------

            for run_observations in runs.values():

                run_observations = sorted(
                    run_observations,
                    key=lambda observation:
                        observation["layer_index"],
                )

                ax.plot(
                    [
                        layer_to_x[
                            observation["layer"]
                        ]
                        for observation
                        in run_observations
                    ],
                    [
                        observation["score"]
                        for observation
                        in run_observations
                    ],
                    color=color,
                    linewidth=0.8,
                    alpha=0.25,
                )

            # -------------------------------------------------
            # Mean profile
            #
            # For "best"/"last":
            #     mean over runs.
            #
            # For "all":
            #     mean over run × epoch observations
            #     at each layer.
            # -------------------------------------------------

            layer_values = defaultdict(list)

            for observation in trainer_observations:

                layer_values[
                    observation["layer"]
                ].append(
                    observation["score"]
                )

            mean_layers = [
                layer
                for layer in layers
                if layer in layer_values
            ]

            mean_scores = [
                np.mean(
                    layer_values[layer]
                )
                for layer in mean_layers
            ]

            ax.plot(
                [
                    layer_to_x[layer]
                    for layer in mean_layers
                ],
                mean_scores,
                color=color,
                linewidth=2.5,
                label=trainer,
                zorder=10,
            )

        # -----------------------------------------------------
        # X-axis
        # -----------------------------------------------------

        ax.set_xticks(
            range(len(layers))
        )

        ax.set_xticklabels(
            layers,
            rotation=60,
            ha="right",
            fontsize=6,
        )

        # -----------------------------------------------------
        # Labels
        # -----------------------------------------------------

        ax.set_title(
            f"{model} {dataset}",
            fontsize=8,
        )

        ax.set_xlabel(
            "Layer",
            fontsize=8,
        )

        ax.set_ylabel(
            "S",
            fontsize=8,
        )

        # -----------------------------------------------------
        # Cosmetics
        # -----------------------------------------------------

        ax.grid(
            axis="y",
            alpha=0.20,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(
            labelsize=7,
        )

    # ---------------------------------------------------------
    # Hide unused panels
    # ---------------------------------------------------------

    for ax in axes[n_panels:]:

        ax.set_visible(False)

    # ---------------------------------------------------------
    # Legend
    # ---------------------------------------------------------

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=trainer_colors["Standard"],
            linewidth=2.5,
            label="Standard",
        ),
        plt.Line2D(
            [0],
            [0],
            color=trainer_colors["Adversarial"],
            linewidth=2.5,
            label="Adversarial",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=0.8,
            alpha=0.4,
            label="Individual run",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------

    checkpoint_label = {
        "best": "Best-S checkpoint",
        "last": "Last checkpoint",
        "all": "All checkpoints",
    }[checkpoint]

    fig.suptitle(
        "Figure 5. Layer-wise interpretability score "
        f"({checkpoint_label})",
        fontsize=10,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    filename = {
        "best": (
            "figure_5_layerwise_score.png"
        ),
        "last": (
            "figure_5_layerwise_score_last.png"
        ),
        "all": (
            "figure_5_layerwise_score_all.png"
        ),
    }[checkpoint]

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved Figure 5 to {filename}"
    )

    plt.close(fig)

def _plot_pairwise_relationship_matrix(
    histories,
    filename="figure_6_pairwise_relationship_matrix.png",
):
    """
    Figure 6:
    Pairwise relationship matrix between

        Accuracy
        PGD Accuracy
        Interpretability Score

    Each point represents one experimental run.
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    rows = []

    for (
        model,
        dataset,
        trainer,
        run,
    ), history in histories.items():

        if not history:
            continue

        snapshot = max(
            history,
            key=lambda s: (
                float("-inf")
                if s.epoch is None
                else s.epoch
            ),
        )

        rows.append(
            {
                "accuracy": snapshot.accuracy,
                "pgd_accuracy": snapshot.pgd_accuracy,
                "interpretability": snapshot.interpretability_score,
                "regime": trainer,
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)

    df = df.dropna(
        subset=[
            "accuracy",
            "pgd_accuracy",
            "interpretability",
        ]
    )

    metrics = [
        "accuracy",
        "pgd_accuracy",
        "interpretability",
    ]

    labels = {
        "accuracy": "Accuracy",
        "pgd_accuracy": "PGD Accuracy",
        "interpretability": "S (Interpretability)",
    }

    regimes = sorted(
        df["regime"].unique()
    )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(9, 9),
        constrained_layout=True,
    )

    for i, y in enumerate(metrics):

        for j, x in enumerate(metrics):

            ax = axes[i, j]

            if i == j:

                ax.hist(
                    df[x],
                    bins=8,
                )

            else:

                for regime in regimes:

                    subset = df[
                        df["regime"] == regime
                    ]

                    ax.scatter(
                        subset[x],
                        subset[y],
                        s=40,
                        alpha=0.8,
                        label=regime,
                    )

            if i == len(metrics)-1:
                ax.set_xlabel(
                    labels[x]
                )

            if j == 0:
                ax.set_ylabel(
                    labels[y]
                )

            ax.grid(
                True,
                alpha=0.3,
            )

    handles, labels_ = axes[0,2].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels_,
            loc="upper right",
            frameon=False,
        )

    fig.suptitle(
        "Pairwise Relationship Matrix",
        fontsize=14,
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return filename

def _figure_7_best_last(
    histories,
    filename="figure_7_best_last.png",
):
    """
    Figure 7:
    Best (circle) vs last (cross) checkpoint comparison.

    Metrics:
        Accuracy
        PGD Accuracy
        Interpretability Score
        Interpretability Integral

    Each horizontal row corresponds to one
    (model, dataset) pairing.

    Lines connect best and last checkpoints.
    """

    import numpy as np
    import matplotlib.pyplot as plt


    metrics = [
        (
            "accuracy",
            "Accuracy",
        ),
        (
            "pgd_accuracy",
            "PGD Accuracy",
        ),
        (
            "interpretability_score",
            "S (Interpretability)",
        ),
        (
            "interpretability_integral",
            "Integral",
        ),
    ]


    rows = []


    # ---------------------------------------------------------
    # Extract best and last snapshots
    # ---------------------------------------------------------

    for (
        model,
        dataset,
        trainer,
        run,
    ), history in histories.items():

        if not history:
            continue


        best = max(
            history,
            key=lambda s: (
                float("-inf")
                if s.interpretability_score is None
                else s.interpretability_score
            ),
        )


        last = max(
            history,
            key=lambda s: (
                float("-inf")
                if s.epoch is None
                else s.epoch
            ),
        )


        rows.append(
            {
                "label": (
                    f"{model}-{dataset}"
                ),

                "trainer": trainer,

                "best": best,

                "last": last,
            }
        )


    if not rows:
        return None


    labels = [
        r["label"]
        for r in rows
    ]


    y = np.arange(
        len(labels)
    )


    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(16, 5),
        sharey=True,
        constrained_layout=True,
    )


    if len(metrics) == 1:
        axes = [axes]


    for ax, (
        metric,
        title,
    ) in zip(
        axes,
        metrics,
    ):

        for idx, row in enumerate(rows):

            best_value = getattr(
                row["best"],
                metric,
                np.nan,
            )

            last_value = getattr(
                row["last"],
                metric,
                np.nan,
            )


            if (
                best_value is None
                or last_value is None
            ):
                continue


            ax.plot(
                [
                    best_value,
                    last_value,
                ],
                [
                    idx,
                    idx,
                ],
                linewidth=1,
                alpha=0.6,
            )


            ax.scatter(
                best_value,
                idx,
                marker="o",
                s=45,
                label="Best",
            )


            ax.scatter(
                last_value,
                idx,
                marker="x",
                s=45,
                label="Last",
            )


        ax.set_title(
            title
        )

        ax.set_xlabel(
            "Value"
        )

        ax.grid(
            True,
            alpha=0.3,
        )


    axes[0].set_yticks(
        y
    )

    axes[0].set_yticklabels(
        labels,
        fontsize=8,
    )


    # ---------------------------------------------------------
    # Remove duplicate legends
    # ---------------------------------------------------------

    handles, labels_ = axes[0].get_legend_handles_labels()

    unique = dict(
        zip(
            labels_,
            handles,
        )
    )

    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=2,
        frameon=False,
    )


    fig.suptitle(
        "Best (●) vs Last (×) checkpoint comparison",
        fontsize=14,
    )


    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return filename

def _figure_8_layerwise_matrix(
    histories,
    filename="figure_8_layerwise_matrix.png",
):
    """
    Figure 8:
    Layer-wise interpretability score matrix.

    Rows:
        experimental runs

    Columns:
        network layers

    Values:
        layer.score
    """

    import numpy as np
    import matplotlib.pyplot as plt


    rows = []
    labels = []


    # ---------------------------------------------------------
    # Extract final checkpoint layer scores
    # ---------------------------------------------------------

    for (
        model,
        dataset,
        trainer,
        run,
    ), history in histories.items():

        if not history:
            continue


        snapshot = max(
            history,
            key=lambda s: (
                float("-inf")
                if s.epoch is None
                else s.epoch
            ),
        )


        scores = [
            layer.score
            for layer in snapshot.layers
        ]


        if not scores:
            continue


        rows.append(scores)

        labels.append(
            f"{model}-{dataset}\n{trainer}-R{run}"
        )


    if not rows:
        return None


    # ---------------------------------------------------------
    # Pad layers if networks have different depth
    # ---------------------------------------------------------

    max_layers = max(
        len(r)
        for r in rows
    )


    matrix = np.full(
        (
            len(rows),
            max_layers,
        ),
        np.nan,
    )


    for i, scores in enumerate(rows):

        matrix[
            i,
            :len(scores)
        ] = scores


    # ---------------------------------------------------------
    # Plot heatmap
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            max(8, max_layers * 0.8),
            max(5, len(labels) * 0.35),
        ),
        constrained_layout=True,
    )


    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )


    ax.set_xlabel(
        "Layer"
    )

    ax.set_ylabel(
        "Experiment"
    )


    ax.set_xticks(
        np.arange(max_layers)
    )

    ax.set_xticklabels(
        [
            f"L{i}"
            for i in range(max_layers)
        ]
    )


    ax.set_yticks(
        np.arange(len(labels))
    )

    ax.set_yticklabels(
        labels,
        fontsize=8,
    )


    plt.colorbar(
        im,
        ax=ax,
        label="Interpretability Score S",
    )


    ax.set_title(
        "Layer-wise Interpretability Matrix",
        fontsize=14,
    )


    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return filename

def _figure_9_width_integral_relationship(
        histories,
        filename="figure_9_width_integral_relationship.png"
):
    """
    Figure 9:
    Layer × width × integral relationship.

    Left:
        Heatmap showing layer-wise integral as a function of
        relative layer width.

    Right:
        Line-slice view showing integral evolution across
        relative width bins.

    The plot uses LayerReport.integral values and layer matrix
    widths when available.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------
    # Collect layer statistics
    # -------------------------------------------------------------
    records = []

    for history in histories:

        snapshots = (
            history.snapshots
            if hasattr(history, "snapshots")
            else history
        )

        for snapshot in snapshots:

            if not hasattr(snapshot, "layers"):
                continue

            layers = snapshot.layers

            n_layers = len(layers)

            for idx, layer_report in enumerate(layers):

                integral = getattr(
                    layer_report,
                    "integral",
                    None
                )

                if integral is None:
                    continue

                # Relative depth position
                depth = (
                    idx / max(n_layers - 1, 1)
                )

                # Estimate relative width
                width = 1.0

                matrix = getattr(
                    layer_report,
                    "matrix",
                    None
                )

                if matrix is not None:
                    try:
                        width = matrix.shape[-1]
                    except Exception:
                        pass

                records.append(
                    {
                        "depth": depth,
                        "layer": idx,
                        "width": width,
                        "integral": float(integral),
                        "model": getattr(
                            snapshot,
                            "model",
                            "unknown"
                        ),
                        "trainer": getattr(
                            snapshot,
                            "trainer",
                            "unknown"
                        ),
                    }
                )

    if len(records) == 0:
        return

    # -------------------------------------------------------------
    # Normalize widths
    # -------------------------------------------------------------
    max_width = max(
        r["width"] for r in records
    )

    for r in records:
        r["relative_width"] = (
            r["width"] / max_width
        )

    widths = np.array(
        [
            r["relative_width"]
            for r in records
        ]
    )

    depths = np.array(
        [
            r["depth"]
            for r in records
        ]
    )

    integrals = np.array(
        [
            r["integral"]
            for r in records
        ]
    )

    # -------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    # =============================================================
    # Left: Heatmap
    # =============================================================

    ax = axes[0]

    width_bins = np.linspace(
        0,
        1,
        20
    )

    depth_bins = np.linspace(
        0,
        1,
        max(
            5,
            len(set(
                int(r["layer"])
                for r in records
            ))
        )
    )

    heatmap = np.zeros(
        (
            len(depth_bins) - 1,
            len(width_bins) - 1
        )
    )

    counts = np.zeros_like(
        heatmap
    )

    for w, d, val in zip(
        widths,
        depths,
        integrals
    ):

        wi = np.searchsorted(
            width_bins,
            w
        ) - 1

        di = np.searchsorted(
            depth_bins,
            d
        ) - 1

        if (
            0 <= di < heatmap.shape[0]
            and 0 <= wi < heatmap.shape[1]
        ):
            heatmap[di, wi] += val
            counts[di, wi] += 1

    counts[counts == 0] = 1

    heatmap /= counts

    im = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        extent=[
            0,
            1,
            0,
            1
        ]
    )

    ax.set_title(
        "Integral heatmap"
    )

    ax.set_xlabel(
        "Relative width"
    )

    ax.set_ylabel(
        "Layer depth"
    )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046
    )


    # =============================================================
    # Right: Line slice view
    # =============================================================

    ax = axes[1]

    depth_groups = np.linspace(
        0,
        1,
        5
    )

    for i in range(
        len(depth_groups) - 1
    ):

        mask = (
            (depths >= depth_groups[i])
            &
            (depths < depth_groups[i + 1])
        )

        if not np.any(mask):
            continue

        x = widths[mask]
        y = integrals[mask]

        order = np.argsort(x)

        ax.plot(
            x[order],
            y[order],
            marker="o",
            label=(
                f"Depth {depth_groups[i]:.2f}-"
                f"{depth_groups[i+1]:.2f}"
            )
        )

    ax.set_title(
        "Line-slice view (by depth bin)"
    )

    ax.set_xlabel(
        "Relative layer width"
    )

    ax.set_ylabel(
        "Integral"
    )

    ax.legend(
        fontsize=8
    )


    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------
    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

def _figure_10_layerwise_score_profile(
        histories,
        filename="figure_10_layerwise_score_profile.png"
):
    """
    Figure 10:
    Layer-wise interpretability score profile.

    x-axis:
        Relative layer depth.

    y-axis:
        Layer score / integral.

    Curves:
        Different training configurations.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict


    # -------------------------------------------------------------
    # Collect layer statistics
    # -------------------------------------------------------------
    profiles = defaultdict(list)


    for history in histories:

        snapshots = (
            history.snapshots
            if hasattr(history, "snapshots")
            else history
        )

        for snapshot in snapshots:

            if not hasattr(snapshot, "layers"):
                continue

            key = (
                getattr(snapshot, "model", "unknown"),
                getattr(snapshot, "trainer", "unknown")
            )

            layers = snapshot.layers

            n_layers = len(layers)

            for idx, layer in enumerate(layers):

                score = getattr(
                    layer,
                    "score",
                    None
                )

                integral = getattr(
                    layer,
                    "integral",
                    None
                )

                if score is None:
                    continue

                depth = (
                    idx / max(n_layers - 1, 1)
                )

                profiles[key].append(
                    {
                        "depth": depth,
                        "score": float(score),
                        "integral": (
                            float(integral)
                            if integral is not None
                            else np.nan
                        )
                    }
                )


    if len(profiles) == 0:
        return


    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharex=True
    )


    ax_score, ax_integral = axes


    for key, values in profiles.items():

        values = sorted(
            values,
            key=lambda x: x["depth"]
        )

        depth = np.array(
            [
                v["depth"]
                for v in values
            ]
        )

        score = np.array(
            [
                v["score"]
                for v in values
            ]
        )

        integral = np.array(
            [
                v["integral"]
                for v in values
            ]
        )


        label = (
            f"{key[0]} / {key[1]}"
        )


        ax_score.plot(
            depth,
            score,
            marker="o",
            label=label
        )


        valid = ~np.isnan(integral)

        if np.any(valid):

            ax_integral.plot(
                depth[valid],
                integral[valid],
                marker="o",
                label=label
            )


    # -------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------
    ax_score.set_title(
        "Layer-wise score profile"
    )

    ax_score.set_xlabel(
        "Relative layer depth"
    )

    ax_score.set_ylabel(
        "Score"
    )


    ax_integral.set_title(
        "Layer-wise integral profile"
    )

    ax_integral.set_xlabel(
        "Relative layer depth"
    )

    ax_integral.set_ylabel(
        "Integral"
    )


    ax_score.legend(
        fontsize=8
    )

    ax_integral.legend(
        fontsize=8
    )


    fig.tight_layout()


    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

def _figure_11_layer_network_aggregation_check(
        histories,
        filename="figure_11_layer_network_aggregation_check.png"
):
    """
    Figure 11:
    Layer-to-network aggregation consistency.

    Whole-network score (reported) versus:
        - mean layer score
        - min layer score
        - max layer score
        - last layer score
    """

    import numpy as np
    import matplotlib.pyplot as plt


    mean_values = []
    min_values = []
    max_values = []
    last_values = []
    network_values = []


    # -------------------------------------------------------------
    # Collect network/layer scores
    # -------------------------------------------------------------
    for history in histories:

        snapshots = (
            history.snapshots
            if hasattr(history, "snapshots")
            else history
        )

        for snapshot in snapshots:

            if not hasattr(snapshot, "layers"):
                continue

            layers = snapshot.layers


            layer_scores = []

            for layer in layers:

                score = getattr(
                    layer,
                    "score",
                    None
                )

                if score is not None:
                    layer_scores.append(
                        float(score)
                    )


            if len(layer_scores) == 0:
                continue


            # -----------------------------------------------------
            # Network score
            # -----------------------------------------------------
            network_score = None


            # possible locations depending on Report version
            for name in [
                "score",
                "network_score",
                "integral"
            ]:
                if hasattr(snapshot, name):
                    value = getattr(
                        snapshot,
                        name
                    )

                    if value is not None:
                        network_score = float(value)
                        break


            if network_score is None:
                continue


            network_values.append(
                network_score
            )

            mean_values.append(
                np.mean(layer_scores)
            )

            min_values.append(
                np.min(layer_scores)
            )

            max_values.append(
                np.max(layer_scores)
            )

            last_values.append(
                layer_scores[-1]
            )


    if len(network_values) == 0:
        return


    network_values = np.asarray(
        network_values
    )


    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(6, 6)
    )


    ax.scatter(
        network_values,
        mean_values,
        marker="o",
        label="mean"
    )

    ax.scatter(
        network_values,
        min_values,
        marker="s",
        label="min"
    )

    ax.scatter(
        network_values,
        max_values,
        marker="^",
        label="max"
    )

    ax.scatter(
        network_values,
        last_values,
        marker="D",
        label="last-layer"
    )


    # identity line
    low = min(
        np.min(network_values),
        np.min(mean_values),
        np.min(min_values),
        np.min(max_values),
        np.min(last_values)
    )

    high = max(
        np.max(network_values),
        np.max(mean_values),
        np.max(min_values),
        np.max(max_values),
        np.max(last_values)
    )


    ax.plot(
        [low, high],
        [low, high],
        linestyle="--"
    )


    ax.set_xlabel(
        "Whole-network S (reported)"
    )

    ax.set_ylabel(
        "Aggregation estimate"
    )


    ax.set_title(
        "Layer-to-network aggregation check"
    )


    ax.legend(
        fontsize=8
    )


    ax.set_xlim(
        low,
        high
    )

    ax.set_ylim(
        low,
        high
    )


    fig.tight_layout()


    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

def _figure_12_layer_importance_contribution(histories):
    """
    Figure 12: Layer importance contribution for STANDARD vs PGD runs.

    Run classification is determined exclusively from:

        snapshot.metadata["trainer"]

    STANDARD runs contribute:

        layer.geometry["standard_contribution"]

    PGD runs contribute:

        layer.geometry["adversarial_contribution"]

    Each run contributes its latest snapshot.

    The plotted values are mean ± 95% CI across runs.

    Output
    ------
    figure_12_layer_importance_contribution.png
    """

    import math
    from collections import defaultdict

    import numpy as np
    import matplotlib.pyplot as plt

    try:
        from scipy.stats import t
        HAVE_SCIPY = True
    except Exception:
        HAVE_SCIPY = False

    # =============================================================
    # Collect layer contributions by training regime
    # =============================================================

    standard_scores = defaultdict(list)
    pgd_scores = defaultdict(list)

    standard_runs = []
    pgd_runs = []

    # =============================================================
    # Process every run
    # =============================================================

    for history_key, history in histories.items():

        if not history:
            continue

        # ---------------------------------------------------------
        # Always retrieve the newest snapshot by epoch.
        # ---------------------------------------------------------

        snapshot = max(
            history,
            key=lambda s: (
                float("-inf")
                if s.epoch is None
                else s.epoch
            )
        )

        # ---------------------------------------------------------
        # Trainer MUST come from metadata.
        #
        # Do not use:
        #
        #     history_key[2]
        #     snapshot.trainer
        #
        # ---------------------------------------------------------

        metadata = getattr(
            snapshot,
            "metadata",
            None,
        )

        if not isinstance(metadata, dict):
            print(
                "Figure 12: skipping run without metadata:",
                history_key,
            )
            continue

        trainer = metadata.get("trainer")

        if trainer is None:
            print(
                "Figure 12: skipping run without metadata['trainer']:",
                history_key,
            )
            continue

        trainer = str(trainer).strip().upper()

        # ---------------------------------------------------------
        # Classify run from metadata.
        # ---------------------------------------------------------

        if trainer in {
            "STANDARD",
            "STD",
        }:

            run_type = "STANDARD"
            standard_runs.append(history_key)

        elif trainer in {
            "PGD",
            "ADVERSARIAL",
            "ADV",
        }:

            run_type = "PGD"
            pgd_runs.append(history_key)

        else:

            print(
                "Figure 12: skipping unknown trainer:",
                trainer,
                "history key:",
                history_key,
            )
            continue

        # ---------------------------------------------------------
        # Extract layer contributions.
        # ---------------------------------------------------------

        layers = getattr(
            snapshot,
            "layers",
            None,
        )

        if layers is None:
            continue

        for layer_idx, layer in enumerate(layers):

            geometry = getattr(
                layer,
                "geometry",
                None,
            )

            if not isinstance(geometry, dict):
                continue

            # -----------------------------------------------------
            # STANDARD run
            # -----------------------------------------------------

            if run_type == "STANDARD":

                value = geometry.get(
                    "standard_contribution"
                )

                if value is None:
                    continue

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue

                if not np.isfinite(value):
                    continue

                standard_scores[
                    layer_idx
                ].append(value)

            # -----------------------------------------------------
            # PGD run
            # -----------------------------------------------------

            elif run_type == "PGD":

                value = geometry.get(
                    "adversarial_contribution"
                )

                if value is None:
                    continue

                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue

                if not np.isfinite(value):
                    continue

                pgd_scores[
                    layer_idx
                ].append(value)

    # =============================================================
    # Diagnostics
    # =============================================================

    print()
    print("Figure 12 run classification")
    print("--------------------------------")

    print(
        "STANDARD runs:",
        len(standard_runs)
    )

    for key in standard_runs:
        print(
            "  STANDARD:",
            key
        )

    print(
        "PGD runs:",
        len(pgd_runs)
    )

    for key in pgd_runs:
        print(
            "  PGD:",
            key
        )

    print()

    # =============================================================
    # Validate that at least one contribution exists
    # =============================================================

    if not standard_scores and not pgd_scores:

        raise ValueError(
            "Figure 12 requires standard/adversarial layer "
            "contribution data.\n\n"
            "Runs are classified using snapshot.metadata['trainer'].\n\n"
            "STANDARD runs require:\n"
            "    layer.geometry['standard_contribution']\n\n"
            "PGD runs require:\n"
            "    layer.geometry['adversarial_contribution']"
        )

    # =============================================================
    # Mean + 95% confidence interval
    # =============================================================

    def mean_ci(values):

        values = np.asarray(
            [
                value
                for value in values
                if value is not None
                and np.isfinite(value)
            ],
            dtype=float,
        )

        if len(values) == 0:
            return np.nan, np.nan

        mean = float(
            np.mean(values)
        )

        if len(values) < 2:
            return mean, np.nan

        std = float(
            np.std(
                values,
                ddof=1,
            )
        )

        sem = std / math.sqrt(
            len(values)
        )

        if HAVE_SCIPY:

            critical = t.ppf(
                0.975,
                len(values) - 1,
            )

        else:

            critical = 1.96

        return (
            mean,
            float(critical * sem),
        )

    # =============================================================
    # Determine layers
    # =============================================================

    all_layers = sorted(
        set(standard_scores.keys())
        |
        set(pgd_scores.keys())
    )

    standard_means = []
    standard_cis = []

    pgd_means = []
    pgd_cis = []

    for layer_idx in all_layers:

        standard_mean, standard_ci = mean_ci(
            standard_scores.get(
                layer_idx,
                [],
            )
        )

        pgd_mean, pgd_ci = mean_ci(
            pgd_scores.get(
                layer_idx,
                [],
            )
        )

        standard_means.append(
            standard_mean
        )

        standard_cis.append(
            0.0
            if np.isnan(standard_ci)
            else standard_ci
        )

        pgd_means.append(
            pgd_mean
        )

        pgd_cis.append(
            0.0
            if np.isnan(pgd_ci)
            else pgd_ci
        )

    # Convert to numpy arrays

    layer_indices = np.asarray(
        all_layers,
        dtype=int,
    )

    standard_means = np.asarray(
        standard_means,
        dtype=float,
    )

    standard_cis = np.asarray(
        standard_cis,
        dtype=float,
    )

    pgd_means = np.asarray(
        pgd_means,
        dtype=float,
    )

    pgd_cis = np.asarray(
        pgd_cis,
        dtype=float,
    )

    # =============================================================
    # Plot
    # =============================================================

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True,
    )

    # -------------------------------------------------------------
    # STANDARD
    # -------------------------------------------------------------

    valid_standard = np.isfinite(
        standard_means
    )

    if np.any(valid_standard):

        x = layer_indices[
            valid_standard
        ]

        y = standard_means[
            valid_standard
        ]

        ci = standard_cis[
            valid_standard
        ]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=5,
            label="STANDARD",
        )

        ax.fill_between(
            x,
            y - ci,
            y + ci,
            alpha=0.15,
        )

    # -------------------------------------------------------------
    # PGD
    # -------------------------------------------------------------

    valid_pgd = np.isfinite(
        pgd_means
    )

    if np.any(valid_pgd):

        x = layer_indices[
            valid_pgd
        ]

        y = pgd_means[
            valid_pgd
        ]

        ci = pgd_cis[
            valid_pgd
        ]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=5,
            label="PGD",
        )

        ax.fill_between(
            x,
            y - ci,
            y + ci,
            alpha=0.15,
        )

    # =============================================================
    # Labels
    # =============================================================

    ax.set_xlabel(
        "Layer"
    )

    ax.set_ylabel(
        "Layer Importance Contribution"
    )

    ax.set_title(
        "Layer Importance Contribution: STANDARD vs PGD"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.set_xticks(
        layer_indices
    )

    ax.legend(
        frameon=False
    )

    # =============================================================
    # Save
    # =============================================================

    output_file = (
        "figure_12_layer_importance_contribution.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved Figure 12 to {output_file}"
    )

def _figure_12_layerwise_score_by_trainer(histories):
    """
    Figure 12: Layer-wise interpretability score by training regime.

    Run classification is determined exclusively from:

        snapshot.metadata["trainer"]

    Supported training-regime labels include:

        STANDARD / STD
        PGD / ADVERSARIAL / ADV

    Each run contributes its latest snapshot.

    For every layer:

        STANDARD -> layer.score
        PGD      -> layer.score

    The plotted values are:

        mean ± 95% CI

    across runs within each training regime.

    Outputs
    -------
    figure_12_layerwise_score_by_trainer.png
    """

    import math
    from collections import defaultdict

    import numpy as np
    import matplotlib.pyplot as plt

    try:
        from scipy.stats import t
        HAVE_SCIPY = True
    except Exception:
        HAVE_SCIPY = False

    # =============================================================
    # Storage
    # =============================================================

    standard_scores = defaultdict(list)
    pgd_scores = defaultdict(list)

    standard_runs = []
    pgd_runs = []

    # =============================================================
    # Process every run
    # =============================================================

    for history_key, history in histories.items():

        if not history:
            continue

        # ---------------------------------------------------------
        # Get latest snapshot by epoch.
        #
        # Do not assume history[-1] is the newest snapshot.
        # ---------------------------------------------------------

        snapshot = max(
            history,
            key=lambda s: (
                float("-inf")
                if s.epoch is None
                else s.epoch
            )
        )

        # ---------------------------------------------------------
        # Trainer MUST come from metadata.
        #
        # Do not use:
        #
        #     history_key[2]
        #     snapshot.trainer
        # ---------------------------------------------------------

        metadata = getattr(
            snapshot,
            "metadata",
            None,
        )

        if not isinstance(metadata, dict):
            print(
                "Figure 12: skipping run without metadata:",
                history_key,
            )
            continue

        trainer = metadata.get(
            "trainer"
        )

        if trainer is None:
            print(
                "Figure 12: skipping run without "
                "metadata['trainer']:",
                history_key,
            )
            continue

        trainer = str(
            trainer
        ).strip().upper()

        # ---------------------------------------------------------
        # Classify the run.
        # ---------------------------------------------------------

        if trainer in {
            "STANDARD",
            "STD",
        }:

            run_type = "STANDARD"

            standard_runs.append(
                history_key
            )

        elif trainer in {
            "PGD",
            "ADVERSARIAL",
            "ADV",
        }:

            run_type = "PGD"

            pgd_runs.append(
                history_key
            )

        else:

            print(
                "Figure 12: skipping unknown trainer:",
                trainer,
                "history key:",
                history_key,
            )

            continue

        # ---------------------------------------------------------
        # Extract layer scores.
        # ---------------------------------------------------------

        layers = getattr(
            snapshot,
            "layers",
            None,
        )

        if layers is None:
            continue

        for layer_idx, layer in enumerate(
            layers
        ):

            score = getattr(
                layer,
                "score",
                None,
            )

            if score is None:
                continue

            try:
                score = float(
                    score
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if not np.isfinite(
                score
            ):
                continue

            # -----------------------------------------------------
            # Store according to metadata trainer.
            # -----------------------------------------------------

            if run_type == "STANDARD":

                standard_scores[
                    layer_idx
                ].append(
                    score
                )

            elif run_type == "PGD":

                pgd_scores[
                    layer_idx
                ].append(
                    score
                )

    # =============================================================
    # Diagnostics
    # =============================================================

    print()
    print(
        "Figure 12 layer-wise score classification"
    )
    print(
        "------------------------------------------"
    )

    print(
        "STANDARD runs:",
        len(standard_runs)
    )

    for key in standard_runs:
        print(
            "  STANDARD:",
            key
        )

    print(
        "PGD runs:",
        len(pgd_runs)
    )

    for key in pgd_runs:
        print(
            "  PGD:",
            key
        )

    print()

    # =============================================================
    # Validate data
    # =============================================================

    if not standard_scores and not pgd_scores:

        raise ValueError(
            "Figure 12 could not find layer scores.\n\n"
            "Runs are classified using "
            "snapshot.metadata['trainer'].\n\n"
            "Expected trainer values include:\n"
            "    STANDARD\n"
            "    PGD\n\n"
            "Each layer must provide:\n"
            "    layer.score"
        )

    # =============================================================
    # Mean + 95% CI
    # =============================================================

    def mean_ci(values):

        values = np.asarray(
            [
                value
                for value in values
                if value is not None
                and np.isfinite(value)
            ],
            dtype=float,
        )

        if len(values) == 0:
            return np.nan, np.nan

        mean = float(
            np.mean(values)
        )

        # No CI possible with a single run.

        if len(values) < 2:
            return mean, np.nan

        std = float(
            np.std(
                values,
                ddof=1,
            )
        )

        sem = std / math.sqrt(
            len(values)
        )

        if HAVE_SCIPY:

            critical = t.ppf(
                0.975,
                len(values) - 1,
            )

        else:

            critical = 1.96

        return (
            mean,
            float(
                critical * sem
            ),
        )

    # =============================================================
    # Determine layer range
    # =============================================================

    all_layers = sorted(
        set(
            standard_scores.keys()
        )
        |
        set(
            pgd_scores.keys()
        )
    )

    layer_indices = np.asarray(
        all_layers,
        dtype=int,
    )

    # =============================================================
    # Calculate STANDARD statistics
    # =============================================================

    standard_means = []
    standard_cis = []

    for layer_idx in all_layers:

        mean, ci = mean_ci(
            standard_scores.get(
                layer_idx,
                [],
            )
        )

        standard_means.append(
            mean
        )

        standard_cis.append(
            0.0
            if np.isnan(ci)
            else ci
        )

    # =============================================================
    # Calculate PGD statistics
    # =============================================================

    pgd_means = []
    pgd_cis = []

    for layer_idx in all_layers:

        mean, ci = mean_ci(
            pgd_scores.get(
                layer_idx,
                [],
            )
        )

        pgd_means.append(
            mean
        )

        pgd_cis.append(
            0.0
            if np.isnan(ci)
            else ci
        )

    standard_means = np.asarray(
        standard_means,
        dtype=float,
    )

    standard_cis = np.asarray(
        standard_cis,
        dtype=float,
    )

    pgd_means = np.asarray(
        pgd_means,
        dtype=float,
    )

    pgd_cis = np.asarray(
        pgd_cis,
        dtype=float,
    )

    # =============================================================
    # Plot
    # =============================================================

    fig, ax = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True,
    )

    # -------------------------------------------------------------
    # STANDARD
    # -------------------------------------------------------------

    valid_standard = np.isfinite(
        standard_means
    )

    if np.any(
        valid_standard
    ):

        x = layer_indices[
            valid_standard
        ]

        y = standard_means[
            valid_standard
        ]

        ci = standard_cis[
            valid_standard
        ]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=5,
            label="STANDARD",
        )

        ax.fill_between(
            x,
            y - ci,
            y + ci,
            alpha=0.15,
        )

    # -------------------------------------------------------------
    # PGD
    # -------------------------------------------------------------

    valid_pgd = np.isfinite(
        pgd_means
    )

    if np.any(
        valid_pgd
    ):

        x = layer_indices[
            valid_pgd
        ]

        y = pgd_means[
            valid_pgd
        ]

        ci = pgd_cis[
            valid_pgd
        ]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=5,
            label="PGD",
        )

        ax.fill_between(
            x,
            y - ci,
            y + ci,
            alpha=0.15,
        )

    # =============================================================
    # Labels
    # =============================================================

    ax.set_xlabel(
        "Layer"
    )

    ax.set_ylabel(
        "Interpretability Score"
    )

    ax.set_title(
        "Layer-wise Interpretability Score: "
        "STANDARD vs PGD"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    if len(
        layer_indices
    ) > 0:

        ax.set_xticks(
            layer_indices
        )

    # Only show legend when at least one
    # regime was successfully plotted.

    if (
        np.any(valid_standard)
        or np.any(valid_pgd)
    ):

        ax.legend(
            frameon=False
        )

    # =============================================================
    # Save
    # =============================================================

    output_file = (
        "figure_12_layerwise_score_by_trainer.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved Figure 12 to {output_file}"
    )