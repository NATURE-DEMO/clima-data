import glob
import os
import zipfile
from collections import defaultdict
from itertools import product
from typing import Any

import cartopy.crs as ccrs
import cdsapi
import pandas as pd  # type: ignore[import-untyped]
import pyproj
import xarray as xr
from xclim.core.dataflags import data_flags

ESGF_NODE = "https://esgf-node.llnl.gov/esg-search"

TIME_FRAMES = {
    "his": {
        "historical": [1976, 2005],
    },
    "rcp": {
        "short": [2011, 2040],
        "medium": [2041, 2070],
        "long": [2071, 2100],
    },
}

r"""Time frames for EURO-CORDEX data

Definition of the time frames used in analysing EURO-CORDEX data and computing climate indices
and climate indicators.
"""

# Scenarios are experiment x period combinations
FUTURE = sum(list(TIME_FRAMES["rcp"].values()), [])
PERIODS = {
    "historical": TIME_FRAMES["his"]["historical"],
    "rcp45": [min(FUTURE), max(FUTURE)],
    "rcp85": [min(FUTURE), max(FUTURE)],
}


class CordexNames:
    r"""Centralized registry for CORDEX climate model naming formats and conversions.

    Handles conversion between different naming conventions used by:
    - Input/keys: Short lowercase keys for easy use
    - ESGF: Used in ESGF table and GCM parts of downloaded filenames
    - CDS: Used in CDS API requests (lowercase with underscores)
    - Filenames: Used in downloaded filenames (hybrid format)
    """

    # EURO-CORDEX coordinate system parameters
    CORDEX_CRS = ccrs.RotatedPole(pole_latitude=39.25, pole_longitude=-162)
    WGS84_TO_CORDEX_TRANS = pyproj.Transformer.from_crs("epsg:4326", CORDEX_CRS)

    # Global Climate Models - mapping from short keys to various formats
    GCMS = {
        "ichec": {
            "input": "ICHEC-EC-EARTH",
            "esgf": "ICHEC-EC-EARTH",
            "cds": "ichec_ec_earth",
        },
        "mpi": {
            "input": "MPI-M-MPI-ESM-LR",
            "esgf": "MPI-M-MPI-ESM-LR",
            "cds": "mpi_m_mpi_esm_lr",
        },
        "cnrm": {
            "input": "CNRM-CERFACS-CM5",
            "esgf": "CNRM-CERFACS-CNRM-CM5",
            "cds": "cnrm_cerfacs_cm5",
        },
        "hadgem": {
            "input": "MOHC-HadGEM2-ES",
            "esgf": "MOHC-HadGEM2-ES",
            "cds": "mohc_hadgem2_es",
        },
    }

    # Regional Climate Models - mapping from short keys to various formats
    RCMS = {
        "racmo": {
            "input": "KNMI-RACMO22E",
            "esgf": "RACMO22E",
            "cds": "knmi_racmo22e",
        },
        "rca4": {
            "input": "SMHI-RCA4",
            "esgf": "RCA4",
            "cds": "smhi_rca4",
        },
        "remo": {
            "input": "GERICS-REMO2015",
            "esgf": "REMO2015",
            "cds": "gerics_remo2015",
        },
        "cosmo": {
            "input": "CLMcom-ETH-COSMO-crCLIM",
            "esgf": "COSMO-crCLIM-v1-1",
            "cds": "clmcom_eth_cosmo_crclim",
        },
    }

    # Variables - mapping from input names to various formats
    VARIABLES = {
        "tas": {
            "input": "tas",
            "esgf": "tas",
            "cds": "2m_air_temperature",
        },
        "sfcWind": {
            "input": "sfcWind",
            "esgf": "sfcWind",
            "cds": "10m_wind_speed",
        },
        "tasmax": {
            "input": "tasmax",
            "esgf": "tasmax",
            "cds": "maximum_2m_temperature_in_the_last_24_hours",
        },
        "tasmin": {
            "input": "tasmin",
            "esgf": "tasmin",
            "cds": "minimum_2m_temperature_in_the_last_24_hours",
        },
        "pr": {
            "input": "pr",
            "esgf": "pr",
            "cds": "mean_precipitation_flux",
        },
        "uas": {
            "input": "uas",
            "esgf": "uas",
            "cds": "10m_u_component_of_the_wind",
        },
        "vas": {
            "input": "vas",
            "esgf": "vas",
            "cds": "10m_v_component_of_the_wind",
        },
        "ta200": {
            "input": "ta200",
            "esgf": "ta200",
            "cds": "200hpa_temperature",
        },
        "ua200": {
            "input": "ua200",
            "esgf": "ua200",
            "cds": "200hpa_u_component_of_the_wind",
        },
        "va200": {
            "input": "va200",
            "esgf": "va200",
            "cds": "200hpa_v_component_of_the_wind",
        },
        "hurs": {
            "input": "hurs",
            "esgf": "hurs",
            "cds": "2m_relative_humidity",
        },
        "huss": {
            "input": "huss",
            "esgf": "huss",
            "cds": "2m_surface_specific_humidity",
        },
        "zg500": {
            "input": "zg500",
            "esgf": "zg500",
            "cds": "500hpa_geopotential_height",
        },
        "ua850": {
            "input": "ua850",
            "esgf": "ua850",
            "cds": "850hpa_u_component_of_the_wind",
        },
        "va850": {
            "input": "va850",
            "esgf": "va850",
            "cds": "850hpa_v_component_of_the_wind",
        },
        "evspsbl": {
            "input": "evspsbl",
            "esgf": "evspsbl",
            "cds": "evaporation",
        },
        "sftlf": {
            "input": "sftlf",
            "esgf": "sftlf",
            "cds": "land_area_fraction",
        },
        "psl": {
            "input": "psl",
            "esgf": "psl",
            "cds": "mean_sea_level_pressure",
        },
        "orog": {
            "input": "orog",
            "esgf": "orog",
            "cds": "orography",
        },
        "ps": {
            "input": "ps",
            "esgf": "ps",
            "cds": "surface_pressure",
        },
        "rsds": {
            "input": "rsds",
            "esgf": "rsds",
            "cds": "surface_solar_radiation_downwards",
        },
        "rlds": {
            "input": "rlds",
            "esgf": "rlds",
            "cds": "surface_thermal_radiation_downward",
        },
        "rsus": {
            "input": "rsus",
            "esgf": "rsus",
            "cds": "surface_upwelling_shortwave_radiation",
        },
        "clt": {
            "input": "clt",
            "esgf": "clt",
            "cds": "total_cloud_cover",
        },
        "mrro": {
            "input": "mrro",
            "esgf": "mrro",
            "cds": "total_run_off_flux",
        },
    }

    # Frequencies - mapping from input names to various formats
    FREQUENCIES = {
        "3hr": {
            "input": "3hr",
            "esgf": "3hr",
            "cds": "3_hours",
        },
        "6hr": {
            "input": "6hr",
            "esgf": "6hr",
            "cds": "6_hours",
        },
        "day": {
            "input": "day",
            "esgf": "day",
            "cds": "daily_mean",
        },
        "mon": {
            "input": "mon",
            "esgf": "mon",
            "cds": "monthly_mean",
        },
        "sea": {
            "input": "sea",
            "esgf": "sea",
            "cds": "seasonal_mean",
        },
    }

    # Experiments - mapping from input names to various formats
    EXPERIMENTS = {
        "historical": {
            "input": "historical",
            "esgf": "historical",
            "cds": "historical",
        },
        "rcp25": {
            "input": "rcp25",
            "esgf": "rcp25",
            "cds": "rcp_2_5",
        },
        "rcp45": {
            "input": "rcp45",
            "esgf": "rcp45",
            "cds": "rcp_4_5",
        },
        "rcp85": {
            "input": "rcp85",
            "esgf": "rcp85",
            "cds": "rcp_8_5",
        },
    }

    @classmethod
    def get_gcm_name(cls, key: str, format: str) -> str:
        """Get GCM name in specified format. Returns input if no match found."""
        if key not in cls.GCMS:
            return key
        if format not in cls.GCMS[key]:
            return key
        return cls.GCMS[key][format]

    @classmethod
    def get_rcm_name(cls, key: str, format: str) -> str:
        """Get RCM name in specified format. Returns input if no match found."""
        if key not in cls.RCMS:
            return key
        if format not in cls.RCMS[key]:
            return key
        return cls.RCMS[key][format]

    @classmethod
    def get_variable_name(cls, name: str, format: str) -> str:
        """Get variable name in specified format. Returns input if no match found."""
        if name not in cls.VARIABLES:
            return name
        if format not in cls.VARIABLES[name]:
            return name
        return cls.VARIABLES[name][format]

    @classmethod
    def get_frequency_name(cls, name: str, format: str) -> str:
        """Get frequency name in specified format. Returns input if no match found."""
        if name not in cls.FREQUENCIES:
            return name
        if format not in cls.FREQUENCIES[name]:
            return name
        return cls.FREQUENCIES[name][format]

    @classmethod
    def get_experiment_name(cls, name: str, format: str) -> str:
        """Get experiment name in specified format. Returns input if no match found."""
        if name not in cls.EXPERIMENTS:
            return name
        if format not in cls.EXPERIMENTS[name]:
            return name
        return cls.EXPERIMENTS[name][format]

    @classmethod
    def get_filename_gcm(cls, key: str) -> str:
        """Get GCM name as it appears in downloaded filenames (ESGF format)."""
        return cls.get_gcm_name(key, "esgf")

    @classmethod
    def get_filename_rcm(cls, key: str) -> str:
        """Get RCM name as it appears in downloaded filenames (input format)."""
        return cls.get_rcm_name(key, "input")

    @classmethod
    def list_gcms(cls) -> list[str]:
        """Get list of all supported GCM keys."""
        return list(cls.GCMS.keys())

    @classmethod
    def list_rcms(cls) -> list[str]:
        """Get list of all supported RCM keys."""
        return list(cls.RCMS.keys())

    @classmethod
    def list_variables(cls) -> list[str]:
        """Get list of all supported variable names."""
        return list(cls.VARIABLES.keys())

    @classmethod
    def list_frequencies(cls) -> list[str]:
        """Get list of all supported frequency names."""
        return list(cls.FREQUENCIES.keys())

    @classmethod
    def list_experiments(cls) -> list[str]:
        """Get list of all supported experiment names."""
        return list(cls.EXPERIMENTS.keys())


def get_dirpath(data_dir: str, variable: str, experiment: str, freq: str) -> str:
    assert variable in CordexNames.VARIABLES, f"Invalid variable name: {variable}"
    assert freq in CordexNames.FREQUENCIES, f"Invalid frequency name: {freq}"
    assert experiment in CordexNames.EXPERIMENTS, f"Invalid experiment name: {experiment}"
    return os.path.join(os.path.expanduser(data_dir), variable, experiment, freq)


def get_files(
    data_dir: str,
    variable: str,
    experiment: str,
    freq: str,
    year_start: int | None = None,
) -> list:
    """Get file paths for the specified variable, experiment, frequency, and year range, for any given RCM model."""
    dir_path = get_dirpath(data_dir, variable, experiment, freq)
    if isinstance(year_start, int):
        year_end = year_start + 4 if freq == "day" else year_start + 9
    elif year_start is None:
        year_end = None
    else:
        raise ValueError("year_start must be an integer or '*'")
    filename = get_filename(variable, "*", "*", experiment, freq, year_start, year_end)
    return glob.glob(dir_path + f"/{filename}.nc")


def get_filename(
    variable: str,
    gcm_model: str,
    rcm_model: str,
    experiment: str,
    freq: str,
    year_start: int | None,
    year_end: int | None,
) -> str:
    """File name for the downloaded data based on the parameters provided

    Return file name following the CDS API naming convention for CORDEX data explained
    at <https://confluence.ecmwf.int/display/CKB/CORDEX%3A+Regional+climate+projections>:

    `<variable>_<domain>_<driving-model>_<experiment>_<ensemble_member>_<rcm-model>_<rcm-run>_<time-frequency>_<temporal-range>.nc`

    Where:
        - <variable> is a short variable name, e.g. “tas” for “temperature at the surface”
        - <domain> is "EUR-11" for EURO-CORDEX data
        - <driving-model> is the GCM model that produced the boundary conditions
        - <experiment> is the name of the experiment used to extract the boundary conditions (historical, rcp45, rcp85)
        - <ensemble-member> is the ensemble identifier in the form “r<X>i<Y>p<Z>”, X, Y and Z are integers
        - <rcm-model> is the name of the model that produced the data
        - <rcm-run> is the version run of the model in the form of "vX" where X is integer
        - <time-frequency> is the time series frequency (e.g., monthly, daily, seasonal)
        - the <temporal-range> is in the form YYYYMM[DDHH]-YYYY[MMDDHH], where Y is year, M is the month, D is day and H is hour. Note that day and hour are optional (indicated by the square brackets) and are only used if needed by the frequency of the data. For example daily data from the 1st of January 1980 to the 31st of December 2010 would be written 19800101-20101231.
    """
    assert variable in CordexNames.VARIABLES, f"Invalid variable name: {variable}"
    domain = "EUR-11"

    # Convert to ESGF format for filename (returns input if not found)
    if gcm_model != "*":
        gcm_model = CordexNames.get_gcm_name(gcm_model, "esgf")

    assert experiment in CordexNames.EXPERIMENTS, f"Invalid experiment name: {experiment}"
    ensemble = "r1i1p1"

    if rcm_model != "*":
        rcm_model = CordexNames.get_rcm_name(rcm_model, "input")

    rcm_run = "v*"

    assert freq in CordexNames.FREQUENCIES, f"Invalid frequency name: {freq}"
    if freq in ["3hr", "6hr", "day"]:
        day_start, day_end = "0101", "1231"
    elif freq == "mon":
        day_start, day_end = "01", "12"
    elif freq == "sea":
        day_start, day_end = "12", "11"
    else:
        raise ValueError(f"Invalid frequency: {freq}")

    if year_start is None:
        year_start = "*"  # type: ignore[assignment]
    if year_end is None:
        year_end = "*"  # type: ignore[assignment]

    return f"{variable}_{domain}_{gcm_model}_{experiment}_{ensemble}_{rcm_model}*_{rcm_run}_{freq}_{year_start}{day_start}-{year_end}{day_end}"


def _check_existing_files(
    dir_path: str,
    variable: str,
    gcm_model: str,
    rcm_model: str,
    experiment: str,
    freq: str,
    year_start: int,
    year_end: int,
) -> list[str]:
    """Check if files matching the download parameters already exist"""
    # Downloaded filenames use:
    # - GCM names in ESGF format (e.g., CNRM-CERFACS-CNRM-CM5)
    # - RCM names in input/full format (e.g., KNMI-RACMO22E, CLMcom-ETH-COSMO-crCLIM-v1-1)
    ensemble = "r1i1p1"  # Default ensemble member

    # Convert to appropriate filename formats
    filename_gcm = CordexNames.get_gcm_name(gcm_model, "esgf")
    filename_rcm = CordexNames.get_rcm_name(rcm_model, "input")

    # Build filename pattern matching the actual downloaded file naming convention
    # Use flexible matching for ensemble member, versions, and dates to handle CDS variations
    pattern = f"{variable}_EUR-11_{filename_gcm}_{experiment}_{ensemble}_{filename_rcm}*_v*_{freq}_{year_start}*-{year_end}*.nc"
    file_pattern = os.path.join(dir_path, pattern)

    match_files = glob.glob(file_pattern)
    return [f for f in match_files if os.path.getsize(f) > 0]


def _build_cds_request(
    variable: str,
    gcm_model: str,
    rcm_model: str,
    experiment: str,
    freq: str,
    year_start: int,
    year_end: int,
) -> dict:
    """Build CDS API request parameters"""
    return {
        "format": "zip",
        "domain": "europe",
        "experiment": CordexNames.get_experiment_name(experiment, "cds"),
        "horizontal_resolution": "0_11_degree_x_0_11_degree",
        "temporal_resolution": CordexNames.get_frequency_name(freq, "cds"),
        "variable": CordexNames.get_variable_name(variable, "cds"),
        "gcm_model": CordexNames.get_gcm_name(gcm_model, "cds"),
        "rcm_model": CordexNames.get_rcm_name(rcm_model, "cds"),
        "ensemble_member": "r1i1p1",
        "start_year": str(year_start),
        "end_year": str(year_end),
    }


def download_cordex(
    variable: str,
    gcm_model: str,
    rcm_model: str,
    experiment: str,
    freq: str,
    year_start: int,
    year_end: int,
    data_dir: str,
    verbose: bool = False,
) -> str:
    """Worker function handling CDS download and file processing"""

    # Check that all parameters are valid
    assert experiment in CordexNames.EXPERIMENTS, f"Invalid experiment name: {experiment}"

    # Configure paths
    filename = get_filename(variable, gcm_model, rcm_model, experiment, freq, year_start, year_end)
    dir_path = get_dirpath(data_dir, variable, experiment, freq)
    zip_path = os.path.join(dir_path, f"{filename.replace('*', 'X')}.zip")
    os.makedirs(dir_path, exist_ok=True)

    # Check if files already exist before downloading
    existing_files = _check_existing_files(
        dir_path, variable, gcm_model, rcm_model, experiment, freq, year_start, year_end
    )
    if existing_files:
        return f"Already exists: {existing_files[0]}. SKIPPING."

    # Create unique client instance per thread
    client = cdsapi.Client()  # url and key are looked up in ~/.cdsapirc

    # Build CDS API request
    request = _build_cds_request(
        variable, gcm_model, rcm_model, experiment, freq, year_start, year_end
    )
    if verbose:
        print(f"Requesting {filename} with parameters: {request}")

    try:
        client.retrieve(
            "projections-cordex-domains-single-levels",
            request,
            zip_path,
        )

        # Process downloaded files
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dir_path)
        os.remove(zip_path)
        return f"Success: {filename}.nc"
    except Exception as e:
        return f"Failed to download {filename}: {e!s}"


# Cache for ESGF data to avoid repeated downloads
_esgf_data_cache = None


def _get_esgf_data() -> Any:
    """Get ESGF data with caching to avoid repeated downloads"""
    global _esgf_data_cache

    if _esgf_data_cache is not None:
        return _esgf_data_cache

    url = "https://raw.githubusercontent.com/euro-cordex/esgf-table/master/euro-cordex-esgf.csv"

    try:
        print("Fetching ESGF table (one-time download)...")
        _esgf_data_cache = pd.read_csv(url)
        print(f"Loaded {len(_esgf_data_cache)} records from ESGF table")
        return _esgf_data_cache
    except Exception as e:
        print(f"Error fetching ESGF data: {e}")
        raise


def get_esgf_combinations_for_experiment(
    experiment: str, variable: str | None = None, frequency: str | None = None
) -> set[tuple[str, str, str, str]]:
    """Get available combinations from ESGF for specific experiment with optional filtering

    Args:
        experiment: Experiment ID (e.g., 'historical', 'rcp45', 'rcp85')
        variable: Optional variable name to filter by
        frequency: Optional frequency to filter by (e.g., 'day', 'mon')

    Returns:
        Set of tuples containing (gcm, rcm, variable, frequency) for valid combinations
    """

    try:
        print(f"Querying ESGF data for experiment: {experiment}")
        if variable:
            print(f"  Filtering by variable: {variable}")
        if frequency:
            print(f"  Filtering by frequency: {frequency}")

        df = _get_esgf_data()

        # Filter by experiment
        filtered_data = df[df["experiment_id"] == experiment]
        if filtered_data.empty:
            print(f"Warning: No data found for experiment '{experiment}' in ESGF table")
            return set()

        # Apply additional filters if provided
        if variable:
            filtered_data = filtered_data[filtered_data["variable"] == variable]
            if filtered_data.empty:
                print(
                    f"Warning: No data found for variable '{variable}' in experiment '{experiment}'"
                )
                return set()

        if frequency:
            filtered_data = filtered_data[filtered_data["frequency"] == frequency]
            if filtered_data.empty:
                print(
                    f"Warning: No data found for frequency '{frequency}' in experiment '{experiment}'"
                )
                return set()

        # Extract unique combinations with all relevant parameters
        combinations = set()
        unique_combinations = filtered_data[
            ["driving_model_id", "model_id", "variable", "frequency"]
        ].drop_duplicates()

        for _, row in unique_combinations.iterrows():
            gcm = row["driving_model_id"]
            rcm = row["model_id"]
            var = row["variable"]
            freq = row["frequency"]
            combinations.add((gcm, rcm, var, freq))

        print(f"Found {len(combinations)} unique combinations for {experiment}")
        return combinations

    except pd.errors.EmptyDataError:
        print("Error: ESGF CSV file is empty or malformed")
        return set()
    except pd.errors.ParserError as e:
        print(f"Error: Could not parse ESGF CSV file: {e}")
        return set()
    except Exception as e:
        print(f"Warning: Could not fetch ESGF data for {experiment}: {e}")
        print("All tasks will be pruned - no combinations available")
        return set()


def prune_invalid_tasks(tasks: list) -> list:
    """
    Prune tasks that are not available in ESGF.

    This function validates GCM-RCM combinations against ESGF availability
    for each experiment using the EURO-CORDEX ESGF table. Note that ESGF
    availability does not guarantee CDS API availability - some combinations
    may exist in ESGF but not be accessible through the CDS API.

    Args:
        tasks: List of task dictionaries with keys: variable, gcm_model, rcm_model, experiment, freq, year_start, year_end

    Returns:
        List of pruned tasks with only valid combinations according to ESGF
    """

    # Group tasks by experiment, variable, and frequency for more efficient ESGF queries
    tasks_by_criteria = defaultdict(list)
    for task in tasks:
        # Get ESGF frequency name (same as input for frequencies)
        esgf_freq = task["freq"]

        criteria = (task["experiment"], task["variable"], esgf_freq)
        tasks_by_criteria[criteria].append(task)

    pruned_tasks = []
    total_pruned = 0

    for (experiment, variable, frequency), exp_tasks in tasks_by_criteria.items():
        print(f"\nProcessing {len(exp_tasks)} tasks for: {experiment}, {variable}, {frequency}")

        # Get available combinations for this specific criteria set
        available_combinations = get_esgf_combinations_for_experiment(
            experiment, variable, frequency
        )

        if not available_combinations:
            print(f"No valid combinations found, skipping all {len(exp_tasks)} tasks")
            total_pruned += len(exp_tasks)
            continue

        exp_pruned, exp_kept = 0, 0

        # Track which combinations we're looking for vs what's available
        requested_combinations = set()

        for task in exp_tasks:
            gcm_norm = CordexNames.get_gcm_name(task["gcm_model"], "esgf")
            rcm_norm = CordexNames.get_rcm_name(task["rcm_model"], "esgf")

            # Create tuple to match the enhanced function's return format
            requested_combo = (gcm_norm, rcm_norm, variable, frequency)
            requested_combinations.add(requested_combo)

            if requested_combo in available_combinations:
                pruned_tasks.append(task)
                exp_kept += 1
            else:
                exp_pruned += 1

        print(f"  Requested {len(requested_combinations)} unique combinations")
        print(f"  Available {len(available_combinations)} combinations in ESGF")
        print(f"  Kept {exp_kept} tasks, pruned {exp_pruned} tasks")

        # Show which combinations were requested but not available
        missing_combinations = requested_combinations - available_combinations
        if missing_combinations:
            print("  Missing combinations:")
            for combo in sorted(missing_combinations):
                print(f"    {combo[0]} + {combo[1]} ({combo[2]}, {combo[3]})")

        total_pruned += exp_pruned

    print(f"ESGF pruning removed {total_pruned} tasks not available in ESGF")
    print(f"Final: {len(pruned_tasks)} valid tasks out of {len(tasks)} original tasks")

    return pruned_tasks


def create_tasks(
    variables: list,
    gcm_list: list | None = None,
    rcm_list: list | None = None,
    periods: dict = PERIODS,
) -> list:
    if gcm_list is None:
        gcm_list = CordexNames.list_gcms()
    if rcm_list is None:
        rcm_list = CordexNames.list_rcms()
    # Define dictionary with listed years for CDS API requests
    years_specs = {
        "day": {  # The data is stored in 5-year files.
            scenario: {
                "start_year": list(range(period[0], period[1] + 1, 5)),
                "end_year": list(range(period[0], period[1] + 1, 5)),
            }
            for scenario, period in periods.items()
        },
        "mon": {  # The data is stored in 10-year files (starting from 2011).
            scenario: {
                "start_year": list(range(period[0], period[1] + 1, 10)),
                "end_year": list(range(period[0], period[1] + 1, 10)),
            }
            for scenario, period in periods.items()
        },
    }

    # Build tasks for all parameter combinations
    tasks = []
    for freq, runs in years_specs.items():
        for experiment, years in runs.items():
            for year_start, year_end in zip(years["start_year"], years["end_year"], strict=False):
                tasks.append({
                    "experiment": experiment,
                    "freq": freq,
                    "year_start": year_start,
                    "year_end": year_end,
                })

    # Build product of variables, GCMs, RCMs and tasks
    all_tasks = [
        {"variable": variable, "gcm_model": gcm, "rcm_model": rcm, **task}
        for variable, gcm, rcm, task in product(variables, gcm_list, rcm_list, tasks)
    ]

    # Apply ESGF-based pruning to remove unavailable combinations
    pruned_tasks = prune_invalid_tasks(all_tasks)

    pruned_count = len(all_tasks) - len(pruned_tasks)
    print(f"Created {len(all_tasks)} initial tasks, pruned {pruned_count} invalid combinations")
    print(
        f"Final: {len(pruned_tasks)} tasks for {len(variables)} variables, {len(gcm_list)} GCMs, {len(rcm_list)} RCMs and {len(years_specs)} time frames"
    )
    return pruned_tasks


def cordex_data_check(filepath: str) -> list:
    """Lightweight quality check for CDS CORDEX data"""
    ds = xr.open_dataset(filepath)

    # Quick data content checks (since metadata is already validated by CDS)
    issues = []
    for var_name, var in ds.data_vars.items():
        # Check for obvious data issues
        if var.isnull().all():
            issues.append(f"{var_name}: All values are NaN")
            continue

        # Use xclim for variable-specific checks
        flags = data_flags(var, ds=ds)
        if flags is not None:
            # flags is a Dataset with boolean scalar values for each check
            failed_checks = []
            for check_name, check_result in flags.data_vars.items():
                if check_result.item():  # .item() extracts the boolean value
                    failed_checks.append(check_name)
            if failed_checks:
                issues.append(f"{var_name}: Failed checks - {', '.join(failed_checks)}")
    ds.close()
    return issues
