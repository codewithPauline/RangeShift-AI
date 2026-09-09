# Real ecological example: GBIF + WorldClim

This example prepares a real species-distribution training table from public biodiversity and climate data. It defaults to the spotted salamander, *Ambystoma maculatum*, in the United States, but the species and country can be changed from the command line.

## Data sources

- **GBIF** occurrence records are retrieved through the public occurrence API after the scientific name is resolved with the current species-matching API.
- **WorldClim 2.1** provides 1970–2000 bioclimatic variables at 10-minute resolution. The example uses BIO1 (annual mean temperature), BIO12 (annual precipitation), and BIO15 (precipitation seasonality).

The script records retrieval time, matched taxon, GBIF key, predictor choices, and source URLs in `provenance.json`.

## Run the example

Install RangeShift with geospatial support:

```bash
pip install -e ".[geo]"
```

Then run:

```bash
python examples/real_ecology/prepare_gbif_worldclim.py \
  --species "Ambystoma maculatum" \
  --country US \
  --max-records 500 \
  --background 500 \
  --output-dir real_ecology_output
```

The workflow will:

1. resolve the scientific name with GBIF;
2. retrieve georeferenced presence records with geospatial-issue filtering;
3. download and cache WorldClim 2.1 bioclimatic rasters;
4. extract BIO1, BIO12, and BIO15 at occurrence coordinates;
5. generate random climate-valid background cells within the observed extent plus a configurable buffer;
6. write `gbif_worldclim_training.csv` and `provenance.json`.

The generated training table can then be passed directly to RangeShift:

```bash
rangeshift calibrate real_ecology_output/gbif_worldclim_training.csv \
  --target presence \
  --features bio1 bio12 bio15 \
  --output spotted_salamander_model.joblib
```

## Scientific boundaries

This example is intentionally reproducible and transparent, but it is **not a publication-ready SDM protocol**. API search results can change over time, opportunistic occurrence records can contain sampling bias, and random background design may not suit every ecological question.

For publication-quality work:

- create a citable GBIF occurrence download with a DOI;
- document taxonomic, temporal, geographic, and basis-of-record filters;
- inspect spatial sampling bias and duplicate locations;
- justify pseudo-absence/background selection;
- consider spatial thinning and target-group background strategies;
- evaluate environmental collinearity and extrapolation;
- use spatially separated validation.

## References

- GBIF API documentation: https://techdocs.gbif.org/en/openapi/
- GBIF occurrence API: https://techdocs.gbif.org/en/openapi/v1/occurrence
- WorldClim 2.1 historical climate: https://www.worldclim.org/data/worldclim21.html
- WorldClim bioclimatic-variable definitions: https://www.worldclim.org/data/bioclim.html
- Fick, S. E. & Hijmans, R. J. (2017). WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas. *International Journal of Climatology*, 37, 4302–4315.
