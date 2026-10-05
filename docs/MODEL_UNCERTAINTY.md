# Model-fit uncertainty (development API)

RangeShift supports two **different** kinds of spread:

- `summarize_suitability_scenarios`: differences among supplied climate scenarios when the fitting assumptions are held fixed.
- `evaluate_model_resampling_uncertainty`: variation in predictions from refitting a balanced Random Forest to different bootstrap training samples while evaluating on the **same held-out rows**.

```python
from rangeshift import evaluate_model_resampling_uncertainty

result = evaluate_model_resampling_uncertainty(
    training_frame,
    evaluation_frame,
    ["bio1", "bio12", "bio15"],
    target_column="presence",
    n_resamples=50,
    random_state=42,
    training_groups=training_frame["spatial_block"].tolist(),
)
result.predictions.to_csv("model_fit_spread.csv", index=True)
result.replicate_metrics.to_csv("replicate_auc.csv", index=False)
```

**Important:** create an independent evaluation set before calling this API. It **does not** split or validate geographic independence automatically; never pass overlapping training/evaluation rows. If observations are spatially autocorrelated, provide block IDs to bootstrap entire blocks. Block resampling is with replacement and may duplicate/omit entire groups. All bootstrap draws must include both target classes. If none can be sampled, an informative error is raised.

The returned 2.5%–97.5% quantiles are descriptive **fit-variability intervals**, *not* confidence or prediction intervals with guaranteed coverage, not calibrated occurrence probabilities, and not uncertainty over future climate models. Default row bootstrap assumes within-class exchangeability and is not appropriate for spatially structured records without justification.

The current API predicts a fixed tabular evaluation set; **raster-wide model-fit uncertainty and crossed climate × model ensembles are planned, not yet implemented**. This feature is post-v0.8 development and does not alter the archived v0.8.0 GitHub release.
