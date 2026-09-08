# RangeShift AI

> Machine-learning tools for predicting species habitat suitability and exploring potential geographic range shifts under environmental change.

[![CI](https://github.com/codewithPauline/RangeShift-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/codewithPauline/RangeShift-AI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

## Why RangeShift AI?

Species ranges are not static. Climate, land use, topography, and other environmental pressures can change where suitable habitat exists. RangeShift AI is being built as a transparent, reproducible machine-learning toolkit for learning how to move from species occurrence/environmental data to habitat-suitability predictions and, ultimately, projected range shifts.

This repository is intentionally developed in stages so that each component is understandable, testable, and scientifically defensible rather than hidden inside a black-box workflow.

## Project status

**Active development — v0.1 foundation.**

The first milestone focuses on one question:

**Can we train and evaluate a reproducible model that predicts current habitat suitability from environmental predictors?**

Current v0.1 capabilities:

- load tabular occurrence/background data;
- validate predictor and target columns;
- split data into training and test sets;
- train a `RandomForestClassifier`;
- report ROC-AUC, accuracy, precision, recall, and F1;
- rank environmental predictors by feature importance;
- save a trained model bundle for reuse;
- run the workflow from Python or the command line;
- test the core training pipeline automatically with GitHub Actions.

## Roadmap

### Phase 1 — Current habitat suitability
- [x] Project architecture
- [x] Baseline Random Forest classifier
- [x] Model evaluation
- [x] Feature importance
- [x] CLI entry point
- [x] Automated tests and CI
- [ ] Real ecological example dataset
- [ ] Spatially aware train/test splitting
- [ ] Probability-based suitability mapping

### Phase 2 — Environmental rasters and mapping
- [ ] Read raster predictor layers
- [ ] Predict suitability across a geographic grid
- [ ] Export GeoTIFF predictions
- [ ] Publication-quality suitability maps

### Phase 3 — Future range-shift projection
- [ ] Future climate/environmental scenario input
- [ ] Current vs. future suitability comparison
- [ ] Stable, gained, and lost habitat classes
- [ ] Range contraction/expansion estimates
- [ ] Geographic centroid movement and shift distance

### Phase 4 — Explainable and robust ML
- [ ] Cross-validation and hyperparameter tuning
- [ ] Spatial block cross-validation
- [ ] Gradient-boosted comparison model
- [ ] SHAP-based model interpretation
- [ ] Calibration and threshold analysis

### Phase 5 — User-facing tool
- [ ] Reproducible end-to-end workflow
- [ ] Interactive visualization interface
- [ ] Documentation/tutorials
- [ ] Packaged release

## Input data format

For v0.1, RangeShift AI expects a CSV containing a binary response column and numeric environmental predictors.

```csv
presence,bio1,bio12,elevation
1,13.2,1050,310
1,14.1,980,270
0,18.9,620,120
0,20.4,540,80
```

`presence` should contain `1` for observed/presence records and `0` for background or absence records.

> **Scientific note:** pseudo-absence/background generation can strongly affect species-distribution models. RangeShift AI does not yet generate these points automatically. That feature will be implemented explicitly rather than silently making ecological assumptions for the user.

## Installation

```bash
git clone https://github.com/codewithPauline/RangeShift-AI.git
cd RangeShift-AI
python -m venv .venv
pip install -e .
```

For development and testing:

```bash
pip install -e ".[dev]"
```

## Command-line example

```bash
rangeshift train data.csv \
  --target presence \
  --features bio1 bio12 elevation \
  --output model.joblib
```

The command prints model-performance metrics and feature importance, then saves the fitted model bundle.

## Python example

```python
import pandas as pd
from rangeshift.model import train_habitat_model

frame = pd.read_csv("data.csv")

result = train_habitat_model(
    frame,
    feature_columns=["bio1", "bio12", "elevation"],
    target_column="presence",
)

print(result.metrics)
print(result.feature_importance)
```

## Repository structure

```text
RangeShift-AI/
├── .github/workflows/     # continuous integration
├── docs/                  # project design and scientific roadmap
├── examples/              # runnable demonstrations
├── src/rangeshift/        # Python package
├── tests/                 # automated tests
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## Design principles

RangeShift AI is being developed around four principles:

1. **Reproducibility** — the same inputs and random seed should reproduce the same baseline result.
2. **Scientific transparency** — ecological assumptions should be explicit.
3. **Software quality** — tested, modular code rather than a single analysis notebook.
4. **Interpretability** — model performance and predictor effects should be inspectable.

## What this project is not

RangeShift AI is not intended to claim that a machine-learning suitability score is automatically equivalent to a realized species range. Dispersal limits, biotic interactions, sampling bias, evolutionary adaptation, land-use change, and extrapolation into novel environmental space can all affect real range shifts. Later versions will make more of these limitations measurable and visible.

## Author

**Pauline Owusu-Ansah**  
Ph.D. researcher in computational and evolutionary biology

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for the full license text.
