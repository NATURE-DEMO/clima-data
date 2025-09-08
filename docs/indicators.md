# Climatic Indicators

Climatic indicators are statistical measures derived from climate variables that quantify specific aspects of climate behavior, trends, and extremes.
These indicators translate raw meteorological data into meaningful metrics for climate impact assessment, risk evaluation and adaptation planning.

The `clima-data` library leverages the [`xclim`](https://xclim.readthedocs.io/) library to compute a comprehensive suite of climatic indicators from CORDEX regional climate projections.
These indicators range from basic statistics (e.g., annual mean temperature) to complex indices that capture extreme events (e.g., extreme precipitation return periods).

## Use in Risk Analysis

Within the **NATURE-DEMO** project framework, these climatic indicators serve as *hazard indices* in the comprehensive risk analysis methodology developed to assess climate threats to critical infrastructure.
By quantifying the frequency, intensity, and duration of climate extremes, these indicators enable the systematic evaluation of climate-related risks and inform the selection of nature-based solutions for infrastructure resilience.

The indicators below are automatically generated from the `clima_data.indices` module, which provides standardized calculations following established climatological practices.

## Required Climate Variables

The following climate variables are automatically recognized by the system when computing indicators:

| Variable | Description | Units |
|----------|-------------|-------|
| `pr` | Mean precipitation flux | kg m⁻² s⁻¹ |
| `tas` | Near-surface air temperature | K |
| `tasmin` | Daily minimum near-surface air temperature | K |
| `tasmax` | Daily maximum near-surface air temperature | K |
| `hurs` | Near-surface relative humidity | % |
| `sfcWind` | Near-surface wind speed | m s⁻¹ |
| `ps` | Surface air pressure | Pa |
| `rsds` | Surface downwelling shortwave radiation | W m⁻² |

For the complete list of available CORDEX variables with detailed descriptions, see the [CORDEX Variables](cordex.md#cordex-variables) section.

## List of climatic indicators

::: indices
