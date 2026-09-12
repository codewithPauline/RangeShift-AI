# Flagship case-study verification

The *Plethodon cinereus* flagship example was executed end to end in GitHub Actions using public GBIF occurrence data, WorldClim 2.1 current climate, and six WorldClim CMIP6 future projections.

The verified workflow completed occurrence/climate preparation, exact-grid raster alignment, six scenario runs, uncertainty summaries, figure generation, provenance collection, and compact result publication successfully.

The repository README and `examples/case_studies/plethodon_cinereus/results/` contain the compact outputs from the verified run. Large downloaded climate rasters remain generated artifacts rather than tracked repository files.

The case-study values are software-demonstration results under the documented assumptions and are not presented as a species conservation forecast.
