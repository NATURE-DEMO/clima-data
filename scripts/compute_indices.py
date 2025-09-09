import gc
import inspect
import logging
from collections.abc import Callable

import xarray as xr
from dask.distributed import Client

import clima_data.indices as indices
from clima_data.cordex import TIME_FRAMES
from clima_data.stats import _check_files_exist, compute_ensemble_stats, ensemble_cat_time
from clima_data.utils import (
    get_common_file_groups,
    get_required_variables,
    validate_realization_alignment,
)

logging.getLogger("distributed.shuffle").setLevel(logging.ERROR)

# Configuration
OVERWRITE = False  # Set to True to overwrite existing files
CORDEX_PATH = "~/data/cordex"
SAVE_PATH = "~/data/clima_data"
FREQ = "day"

# Indices to compute - set to None to compute all, or provide list of specific indices
# Examples:
#  SELECTED_INDICES = None  # Compute all available indices
#  SELECTED_INDICES = ["solidprcptot_winter", "solidprcptot_year"]  # Snow indices only
#  SELECTED_INDICES = ["rx1day", "cdd", "cwd"]  # Precipitation extremes only
#  SELECTED_INDICES = ["tg_mean_year", "tx_mean_year", "tn_mean_year"]  # Temperature means only
SELECTED_INDICES = ["par_plant_level", "spei3_severe_prob"]
# SELECTED_INDICES = None

# Dask configuration
DASK_CONFIG = {
    "n_workers": 32,
    "threads_per_worker": 1,
    "memory_limit": "10GB",  # request 320GB for 32 workers
    "local_directory": "/tmp",  # Use local tmp for faster I/O on cluster
}

CHUNK_CONFIG = {
    "time": 1000,  # Smaller time chunks for better memory management
    "x": 100,  # Smaller spatial chunks to reduce memory per task
    "y": 100,  # Smaller spatial chunks to reduce memory per task
    "realization": 2,  # Fewer realizations per chunk for memory efficiency
}


def show_available_indices() -> None:
    """Display all available climate indices."""
    all_indices = [
        name
        for name, func in inspect.getmembers(indices, inspect.isfunction)
        if func.__module__ == indices.__name__
    ]
    print(f"\nAvailable indices ({len(all_indices)}):")
    for idx in sorted(all_indices):
        print(f"  - {idx}")
    print()


def get_indicator_functions() -> dict[frozenset, list[tuple[str, Callable]]]:
    """Get indicator functions grouped by required variable sets."""
    # Get all available indicator functions
    all_indicator_functions = [
        (name, func, get_required_variables(func))
        for name, func in inspect.getmembers(indices, inspect.isfunction)
        if func.__module__ == indices.__name__
    ]

    # Filter based on SELECTED_INDICES if specified
    if SELECTED_INDICES is not None:
        indicator_functions = [
            (name, func, variables)
            for name, func, variables in all_indicator_functions
            if name in SELECTED_INDICES
        ]
        print(f"Filtering to selected indices: {SELECTED_INDICES}")
        missing_indices = set(SELECTED_INDICES) - {name for name, _, _ in indicator_functions}
        if missing_indices:
            print(f"Warning: Selected indices not found: {missing_indices}")
    else:
        indicator_functions = all_indicator_functions
        print("Computing all available indices")

    indicators_by_variables: dict[frozenset, list[tuple[str, Callable]]] = {}
    for indicator_name, indicator_fun, indicator_variables in indicator_functions:
        print(f"Indicator: {indicator_name}, Required Variables: {indicator_variables}")
        # Group by variable set (convert to frozenset for hashing)
        variable_set = frozenset(indicator_variables)
        if variable_set not in indicators_by_variables:
            indicators_by_variables[variable_set] = []
        indicators_by_variables[variable_set].append((indicator_name, indicator_fun))

    print("\nGrouped indicators by variable sets:")
    for variable_set, indicators in indicators_by_variables.items():
        indicator_names = [name for name, _ in indicators]
        variables_str = ", ".join(sorted(variable_set))
        print(f"  {variables_str}: {indicator_names}")

    return indicators_by_variables


def load_and_process_ensemble(
    variables: list[str],
    exp: str,
    year_start: int,
    year_end: int,
    indicators: list[tuple[str, Callable]] = None,
    cordex_path: str = CORDEX_PATH,
) -> xr.Dataset:
    """Load ensemble data for multiple variables and time period with guaranteed realization alignment."""
    # Get file groups that exist for ALL variables with consistent model ordering
    print(f"Finding common models across variables {variables}...")
    file_groups_per_var, model_metadata = get_common_file_groups(
        variables=variables,
        cordex_path=cordex_path,
        experiment=exp,
        frequency=FREQ,
        year_start=year_start,
        year_end=year_end,
    )

    print(f"Found {len(model_metadata)} common models across all variables:")
    for i, meta in enumerate(model_metadata):
        print(f"  {i}: {meta['model_id']} ({meta['gcm']} -> {meta['rcm']})")

    chunk_config = CHUNK_CONFIG.copy()
    chunk_config["realization"] = max(chunk_config["realization"] // len(variables), 1)

    # For memory-intensive indices like SPEI, use even smaller chunks
    if indicators:
        memory_intensive_indices = ["spei3_severe_prob"]
        current_indices = [name for name, _ in indicators]

        if any(idx in memory_intensive_indices for idx in current_indices):
            chunk_config["x"] = 75  # Reasonable chunk size - not too small
            chunk_config["y"] = 75  # Reasonable chunk size - not too small
            chunk_config["realization"] = 1  # Process one realization at a time
            print(f"Using balanced chunks for memory-intensive indices: {chunk_config}")

    # Load ensemble data for each variable using aligned file groups
    data_arrays = []
    for i, variable in enumerate(variables):
        print(f"Loading ensemble data for variable: {variable} [{i + 1} of {len(variables)}]")

        ens_da = ensemble_cat_time(
            file_groups=file_groups_per_var[i],
            variable=variable,
            apply_fix=True,
            project_id="cordex",
            model_metadata=model_metadata,
            chunk_config=chunk_config,
        )
        data_arrays.append(ens_da)

    # Convert to datasets and validate realization alignment before merging
    print("Validating realization alignment...")
    datasets = [da.to_dataset() for da in data_arrays]
    validate_realization_alignment(datasets)

    # Combine variables into Dataset with strict coordinate checking
    print(f"Combining variables {variables} into ensemble Dataset...")
    ensemble_dataset = xr.merge(datasets, compat="identical")
    ensemble_dataset = ensemble_dataset.persist()

    return ensemble_dataset


def process_indicators(
    ensemble_data: xr.Dataset,
    variables: list[str],
    indicators: list[tuple[str, Callable]],
    label: str,
    overwrite: bool = False,
) -> None:
    """Process all indicators for given ensemble data."""
    variables_str = ", ".join(sorted(variables))
    print(f"Computing {len(indicators)} indicators for variables [{variables_str}]...")

    for indicator_name, indicator_fun in indicators:
        try:
            print(f" - Running {indicator_name}_{label}")

            # Always pass the Dataset - compute_ensemble_stats will handle single vs multi-variable
            compute_ensemble_stats(
                data=ensemble_data,
                indicator_fun=indicator_fun,
                save_path=SAVE_PATH,
                label=label,
                overwrite=overwrite,
            )

        except Exception as e:
            print(f"    Error computing {indicator_name}: {e}")
            # Force garbage collection after failed computation
            gc.collect()


def main() -> None:
    """Main processing loop."""
    # Show available indices for reference
    show_available_indices()

    # Get the indicators to compute (filtered or all)
    indicators_by_variables = get_indicator_functions()

    try:
        for exp in ["historical", "rcp45", "rcp85"]:
            for time_frame, (year_start, year_end) in TIME_FRAMES[exp[:3]].items():
                print(f"\n=== Processing experiment: {exp}, time frame: {time_frame} ===")
                label = f"{exp}_{time_frame}" if exp != "historical" else "historical"

                for variable_set, indicators in indicators_by_variables.items():
                    # Check if any indicator needs to be computed or if all files already exist
                    exist_file = [
                        _check_files_exist(indicator_fun, label, SAVE_PATH)
                        for _, indicator_fun in indicators
                    ]
                    if not OVERWRITE and all(exist_file):
                        print("    All indicators already computed. Skipping...")
                        continue

                    variables = list(variable_set)
                    variables_str = ", ".join(sorted(variables))
                    print(
                        f"\nLoading data for variables: [{variables_str}], experiment: {exp}, time frame: {time_frame}"
                    )
                    ensemble_data = load_and_process_ensemble(
                        variables, exp, year_start, year_end, indicators
                    )
                    process_indicators(
                        ensemble_data, variables, indicators, label, overwrite=OVERWRITE
                    )

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Cleaning up resources...")
        gc.collect()


if __name__ == "__main__":
    with Client(**DASK_CONFIG) as client:
        print(f"Dask client dashboard available at: {client.dashboard_link}")
        main()
    print("DONE COMPUTING INDICES.")
