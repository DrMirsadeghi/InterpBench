# InterpBench

**InterpBench** is a benchmarking framework for deep neural networks that unifies **generalization**, **robustness**, **interpretability**, and **representation geometry** into a single evaluation pipeline.

Inspired by RobustBench, InterpBench provides a single, opinionated benchmark with a minimal public API.

---

## Philosophy

InterpBench is **not** an interpretability toolbox.

It is a **benchmarking framework**.

The benchmark specification—including interpretability methods, robustness evaluation, and geometry analysis—is built into the library. Users do not configure individual explainers or metrics; they benchmark models.

---

# Installation

```bash
pip install interpbench
```

---

# Public API

The library consists of three pillars.

## 1. Evaluate

Benchmark a trained PyTorch model.

```python
import interpbench as ib

report = ib.evaluate(model, testloader)
```

The result is a canonical `Report` object.

---

## 2. Trainers

Train while automatically benchmarking every checkpoint.

```python
trainer = ib.StandardTrainer(
    model,
    trainloader,
    testloader,
)

trainer.fit(100)

report = trainer.report
```

Adversarial training is identical.

```python
trainer = ib.PGDTrainer(...)
trainer.fit(100)

report = trainer.report
```

Additional trainers follow the same interface.

---

## 3. Plot

Visualize any benchmark report.

```python
ib.plot(report)
```

The plotting engine automatically determines the structure of the report and generates appropriate visualizations.

---

# Report

The `Report` is the canonical object of InterpBench.

Every operation in the library either

- produces a Report,
- consumes a Report, or
- transforms a Report.

Reports are pickle-serializable and can be saved, loaded, merged, filtered, or extended.

```python
report = ib.evaluate(...)

with open("experiment.pkl", "wb") as f:
    pickle.dump(report, f)
```

---

# Report Hierarchy

A Report is a hierarchical collection.

```
Report
 ├── Snapshot
 ├── Snapshot
 ├── Snapshot
 └── ...
```

A report may represent

- a single evaluated model,
- an entire training trajectory,
- multiple training trajectories,
- or an arbitrary collection of experiments.

The public API does not distinguish between these cases.

---

# Snapshot

A Snapshot represents one neural network at one point in time.

Each snapshot contains the complete benchmark output, including

- model metadata
- training metadata
- generalization metrics
- robustness metrics
- interpretability metrics
- hull geometry
- benchmark metadata

Snapshots are implementation details contained within a Report.

---

# Automatic Benchmarking

During training, InterpBench repeatedly evaluates the model.

```
Train Epoch
      ↓
Checkpoint
      ↓
Benchmark
      ↓
Append Snapshot
      ↓
Continue Training
```

At the end of training, the trainer exposes a single Report containing the complete benchmark history.

---

# Design Principles

- Minimal public API
- Opinionated benchmark specification
- One canonical output object
- Fully reproducible evaluation
- Automatic benchmarking during training
- Native support for robustness, interpretability, and geometry analysis

---

# Example

```python
import interpbench as ib

# Benchmark a pretrained model
report = ib.evaluate(model, testloader)

# Train while benchmarking
trainer = ib.PGDTrainer(
    model,
    trainloader,
    testloader,
)

trainer.fit(100)

report = trainer.report

# Visualize the results
ib.plot(report)
```