# CORDEX Regional Climate Projections

## Climate Models and Climatic Indices

Climate models are sophisticated computational tools that simulate Earth's climate system by integrating physical, chemical, and biological processes across the atmosphere, oceans, land, and ice.
These models, ranging from global (GCMs) to regional (RCMs) scales, use mathematical equations to represent interactions such as energy transfer, ocean currents, and greenhouse gas effects.
By projecting future scenarios under varying emissions pathways (e.g., RCPs or SSPs), they generate data on key variables like temperature, precipitation, and wind.
These outputs feed into climatic indices—metrics such as heatwave frequency, drought severity, or extreme rainfall return periods—that quantify specific climate risks.

CORDEX provides high-resolution regional climate projections essential for assessing localized climate impacts across 14 continent-scale domains, including Europe [@Diez-Sierra2022].
The climatic indices provided by such projections are vital for policymakers, guiding adaptation strategies and resilience planning in sectors like agriculture, hydrology, and urban development.

## Global Climate Models (GCMs)

Several studies have evaluated and ranked CMIP5 Global Climate Models (GCMs) used in EURO-CORDEX based on their ability to simulate historical climate and provide reliable boundary conditions for regional downscaling.
Based on [@Jury2015], we select a subset of GCMs as top performers for EURO-CORDEX.

## Regional Climate Models (RCMs)

Based on evaluations from the EURO-CORDEX ensemble and peer-reviewed studies, there is no universal "best" regional climate model (RCM), but consensus exists on top-performing models for specific variables and regions [@Kotlarski2015; @Coppola2021].

## CORDEX Variables

The table below provides a partial list of variables available in EURO-CORDEX — such as near-surface temperature (`tas`), precipitation (`pr`), and wind speed (`sfcWind`).
The complete list is available in the [CORDEX Variable Requirements](https://is-enes-data.github.io/CORDEX_variables_requirement_table.pdf) document.
These variables serve as foundational inputs for deriving critical climatic indices, ranging from heatwave duration (`tx40_cdd`) to extreme precipitation return periods (`r100yrRP`), which quantify climate extremes and trends, enabling researchers and policymakers to evaluate risks like droughts, floods, and temperature anomalies.
By leveraging these standardized variables, stakeholders can perform robust, region-specific analyses to inform adaptation strategies and resilience planning under evolving climate scenarios.

{{ read_csv('./assets/cordex_variables.csv') }}

### Notes
- **Units**: Align with CORDEX/CDS specifications (e.g., `pr` in `kg.m⁻².s⁻¹`, `tas` in `K`)
- **Temporal resolution**: Variables are available at 3-hourly, daily, monthly, or seasonal frequencies, which in the CDS API are indicated as `3hr`, `daily_mean`, `monthly_mean`, `seasonal_mean`, respectively (non-European domains only include daily data).
- **Static variables**: `sftlf` (land area fraction) and `orog` (topography) are time-independent
- **Standard Names**: Standard naming following CF conventions can be found in [CORDEX Variable Requirements Table](https://is-enes-data.github.io/CORDEX_variables_requirement_table.pdf)

For further details, see the [CORDEX Documentation](https://confluence.ecmwf.int/display/CKB/CORDEX%3A+Regional+climate+projections).

## Downloading CORDEX data

To download all the CORDEX data necessary to compute climatic indices and indicators useful to NATURE-DEMO, make sure to set up a Copernicus Climate Data Store (CDS) account as explained in the [main page](index.md#setup).
Then run the following script:

```bash
python scripts/download_cordex_data.py
```

This will take a long time (~24h), since it will download a lot of data (~5TB).
The data will be downloaded to the folder `~/data/cordex`.
That path can be changed by modifying the `DATADIR` variable in the script.

This download script uses the code in the module `clima_data.cordex`, which provides a convenient interface to the Copernicus Climate Data Store (CDS) API for downloading CORDEX data and is documented below.

## Code documentation `clima_data.cordex`

::: cordex
