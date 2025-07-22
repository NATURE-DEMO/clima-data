# Welcome to clima-data

`clima-data` is a Python library and collection of scripts designed for the [**NATURE-DEMO** project](https://nature-demo.eu) as part of Work Packeage 2 (WP2).
It provides tools for downloading and processing CORDEX (Coordinated Regional Climate Downscaling Experiment) data and calculating climate indices from regional climate model data.

This documentation provides a guide to using the library, understanding the data it relies on, and exploring the climatic indicators it can generate.

## Key Features

*   **Climate Data Access**: Seamlessly download and process high-resolution regional climate projections from the [CORDEX](cordex.md) project.
*   **Climatic Indicators**: Calculate a wide range of [climatic indicators](indicators.md) using the robust `xclim` library.
*   **Extensible and Modular**: The library is designed to be easily extended with new climate models, regions, and indicators.

## Documentation Overview

*   **[CORDEX Climate Data](cordex.md)**: Learn about the CORDEX regional climate projections, including the available models, variables, and how to download the data.
*   **[Climatic Indicators](indicators.md)**: Explore the available climatic indicators, their definitions, and how to compute them using the library.

## Getting Started

To get started with `clima-data`, you'll need to set up your environment and download the necessary climate data.

### Setup

1.  **Copernicus CDS Account**: You'll need a Copernicus Climate Data Store (CDS) account to download CORDEX data. If you don't have one, you can sign up for free at the [CDS website](https://cds.climate.copernicus.eu).
2.  **API Key**: Once you have an account, find your API key on your [profile page](https://cds.climate.copernicus.eu/profile) and create a `.cdsapirc` file in your home directory with your credentials.
3.  **Installation**: Clone this repository and install the necessary dependencies using `make install`.

For detailed setup instructions, please see the [CORDEX Climate Data](cordex.md#downloading-cordex-data) page.

## Project Acknowledgment

This work is part of the **NATURE-DEMO** project, funded by the European Union's Horizon Europe research and innovation programme under grant agreement No. 101157448. 

NATURE-DEMO integrates nature-based solutions to enhance climate resilience for critical infrastructure across Europe.
