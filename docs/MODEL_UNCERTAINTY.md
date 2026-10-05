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

The tabular API predicts a fixed independent evaluation set. The additional `summarize_model_fit_rasters` API takes independently trained and scientifically comparable model bundles, then produces windowed GeoTIFFs for model-fit mean, sample SD, 2.5%/97.5% descriptive quantiles, and optional shared-threshold agreement. The caller must supply truly independent training fits (for example, fits generated using spatial-block resampling); the function does **not** train or validate those models itself. A crossed climate × model ensemble is not implemented. These features are post-v0.8 development and does not alter the archived v0.8.0 GitHub release.

## Train and map an ensemble in one call

```python
from rangeshift import run_bootstrap_raster_uncertainty

outputs = run_bootstrap_raster_uncertainty(
    training_frame,
    ["bio1", "bio12", "bio15"],
    {"bio1": "bio1.tif", "bio12": "bio12.tif", "bio15": "bio15.tif"},
    "outputs/model_fit_uncertainty",
    training_groups=training_frame["spatial_block"].tolist(),
    n_resamples=30,
    random_state=42,
    threshold=validated_threshold,
)
print(outputs.manifest_path)
print(outputs.raster.sd_path)
```

Use spatial groups for geographically structured samples; these must be specified by the researcher before bootstrap fitting. This workflow fits class-balanced Random Forests without an extra calibration step, so the resulting scores should not be interpreted as empirically calibrated occupancy probabilities. The optional shared threshold must have been selected on a **separate validation set**; do not tune it on the projected raster. The manifest records seed, predictor order, bootstrap sizes and raster paths but does **not** snapshot the original data files or save fitted models. Compare the same input grid across resamples for meaningful variation.
