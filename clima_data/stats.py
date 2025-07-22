import glob
import inspect
import os
from collections.abc import Callable

import xarray as xr
from c3s_atlas.fixers import (
    adding_coords,
    fix_360_longitudes,
    fix_inverse_latitudes,
    fix_spatial_coord_names,
    fix_time,
    rename_and_delete_variables,
    reorder_dimensions,
    resampled_by_temporal_aggregation,
    standard_names,
)

from clima_data.utils import align_coordinates, get_required_variables


def fix_data_variable_standard_names(ds: xr.Dataset) -> xr.Dataset:
    """Fix standard names for climate data variables that c3s_atlas.fixers.standard_names misses."""
    # Define correct standard names for common climate variables
    variable_standard_names = {
        "tas": "air_temperature",
        "hurs": "relative_humidity",
        "pr": "precipitation_flux",
        "tasmax": "air_temperature",
        "tasmin": "air_temperature",
        "psl": "air_pressure_at_mean_sea_level",
        "sfcWind": "wind_speed",
        "rsds": "surface_downwelling_shortwave_flux_in_air",
    }

    # Apply standard names to data variables that exist in the dataset
    print(f"  - Fixing standard names for variables: {list(ds.data_vars)}")
    for var_name, standard_name in variable_standard_names.items():
        if var_name in ds.data_vars:
            old_standard_name = ds[var_name].attrs.get("standard_name", "None")
            ds[var_name].attrs["standard_name"] = standard_name
            print(f"    {var_name}: '{old_standard_name}' -> '{standard_name}'")

    return ds


def ensemble_cat_time(
    file_groups: list[list[str]],
    variable: str,
    apply_fix: bool = True,
    project_id: str = "cordex",
    model_metadata: list[dict[str, str]] | None = None,
    chunk_config: dict[str, int] | None = None,
) -> xr.DataArray:
    """Concatenate and Ensemble Datasets

    Concatenates along the `time` dimension multiple xarray datasets from a list of file paths into a single
    dataset grouped by their base filenames, and then creates an ensemble from these datasets.
    e.g. [[ 'pr_EUR-11_ICHEC-EC-EARTH_historical_r1i1p1_KNMI-RACMO22E_v1_day_19860101-19901231.nc',
            'pr_EUR-11_ICHEC-EC-EARTH_historical_r1i1p1_KNMI-RACMO22E_v1_day_19810101-19851231.nc',
            'pr_EUR-11_ICHEC-EC-EARTH_historical_r1i1p1_KNMI-RACMO22E_v1_day_19760101-19801231.nc'],
            ['pr_EUR-11_ICHEC-EC-EARTH_historical_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_day_19760101-19801231.nc',
            'pr_EUR-11_ICHEC-EC-EARTH_historical_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_day_19860101-19901231.nc',
            'pr_EUR-11_ICHEC-EC-EARTH_historical_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_day_19810101-19851231.nc']]

    Args:
        file_groups: List of lists, where each sublist contains file paths to datasets that should be concatenated together along time,
        variable: str, the variable name to extract from the datasets
        apply_fix: bool, whether to apply the cs3_atlas fixers to the datasets
        project_id: str, the project identifier, e.g. 'cordex'
    """
    # Default chunking strategy if not provided
    if chunk_config is None:
        chunk_config = {"time": 2000, "x": 200, "y": 200}

    # Extract realization chunk size if provided, default to 1
    realization_chunks = chunk_config.get("realization", 1)

    # Create file chunks without realization dimension for file loading
    file_chunks = {k: v for k, v in chunk_config.items() if k != "realization"}

    ens = [
        xr.open_mfdataset(
            g,
            combine="nested",
            concat_dim="time",
            parallel=True,
            engine="h5netcdf",
            chunks=file_chunks,
        )
        for g in file_groups
    ]

    if apply_fix:
        print("- Applying fixers to datasets...")
        map_variables = {
            "dataset_variable": {variable: "data"},
            "aggregation": {"data": "mean"},
        }

        for n in range(len(ens)):
            ens[n] = fix_spatial_coord_names(ens[n])
            ens[n] = fix_time(ens[n])
            ens[n] = rename_and_delete_variables(ens[n], variable, map_variables)
            ens[n] = fix_360_longitudes(ens[n], project=project_id)
            ens[n] = fix_inverse_latitudes(ens[n], project=project_id)
            ens[n] = resampled_by_temporal_aggregation(ens[n], map_variables)
            ens[n] = reorder_dimensions(ens[n])
            ens[n] = adding_coords(ens[n])
            ens[n] = standard_names(ens[n])
            ens[n] = fix_data_variable_standard_names(ens[n])

        # Re-chunk after fixes to maintain chunking
        for n in range(len(ens)):
            ens[n] = ens[n].chunk(file_chunks)

    # Extract the variable of interest as DataArray from each fixed Dataset
    data_arrays = []
    for dataset in ens:
        if variable not in dataset.data_vars:
            print(
                f"Warning: Variable '{variable}' not found in dataset. Available variables: {list(dataset.data_vars)}"
            )
            print(f"SKIPPING file: {dataset}")
        else:
            data_arrays.append(dataset[variable])

    # align_coordinates returns List[DataArray], so concat returns DataArray
    print(f"- Concatenating {len(data_arrays)} DataArrays along 'realization' dimension")
    ensemble_da = xr.concat(align_coordinates(data_arrays), dim="realization")

    # Add model metadata to realization coordinate if provided
    if model_metadata:
        if len(model_metadata) != len(data_arrays):
            raise ValueError(
                f"Model metadata length ({len(model_metadata)}) doesn't match data arrays ({len(data_arrays)})"
            )

        # Create model_id coordinate with model identifiers
        model_ids = [meta["model_id"] for meta in model_metadata]
        ensemble_da = ensemble_da.assign_coords(realization=model_ids)

        # Add model metadata as attributes
        ensemble_da.attrs["model_metadata"] = model_metadata

    # Ensure final result maintains optimal chunking
    final_chunks = {
        **file_chunks,
        "realization": realization_chunks,
    }
    ensemble_da = ensemble_da.chunk(final_chunks)

    return ensemble_da


def _ensemble_stats_aggregation_fun(da: xr.DataArray) -> xr.DataArray:
    """
    Compute ensemble statistics: mean, variance, and count.
    Returns DataArray with statistic dimension.
    """
    # First handle time dimension if present
    if "time" in da.dims:
        da = da.mean(dim="time")

    # Compute ensemble statistics
    ensemble_mean = da.mean(dim="realization")
    ensemble_var = da.var(dim="realization")
    ensemble_count = da.count(dim="realization")

    # Combine into single DataArray with statistics dimension
    stats_data = xr.concat([ensemble_mean, ensemble_var, ensemble_count], dim="statistic")

    # Add coordinate labels
    stats_data.coords["statistic"] = ["mean", "variance", "count"]

    # Preserve attributes
    stats_data.attrs = da.attrs.copy()
    stats_data.attrs["ensemble_statistics"] = "mean, variance, count"

    return stats_data


def _save_data_array(data: xr.DataArray, filepath: str, overwrite: bool) -> bool:
    """Save DataArray to file, handling overwrite logic."""
    if not overwrite and os.path.exists(filepath):
        print(f"      File {filepath} already exists. Skipping.")
        return False

    print(f"      Saving {filepath}")
    data.to_netcdf(filepath, format="NETCDF4")
    return True


def _build_filename(indicator_fun: Callable, label: str | None) -> str:
    """Build filename from indicator function name and label."""
    filename = getattr(indicator_fun, "__name__", "indicator")
    # Add label if provided
    if label:
        filename += f"_{label}"
    return filename


def _check_files_exist(indicator_fun: Callable, label: str | None, save_path: str) -> bool:
    """Check if output files already exist. Returns True if files exist and shouldn't be overwritten."""
    save_path = os.path.expanduser(save_path)
    indicator_name = getattr(indicator_fun, "__name__", "indicator")

    # Check for return period files if this is a return period indicator
    if indicator_name.endswith("_rp"):
        # Return period indicators: rx1day_rp2_historical.nc, rx1day_rp10_historical.nc
        if label:
            pattern = os.path.join(save_path, f"{indicator_name}[0-9]*_{label}.nc")
        else:
            pattern = os.path.join(save_path, f"{indicator_name}[0-9]*.nc")
    else:
        # Standard indicators: rx1day_historical.nc
        base_filename = _build_filename(indicator_fun, label)
        pattern = os.path.join(save_path, f"{base_filename}.nc")

    existing_files = glob.glob(pattern)
    return bool(existing_files)


def compute_ensemble_stats(
    data: xr.Dataset,
    indicator_fun: Callable,
    indicator_kwargs: dict | None = None,
    aggregation_fun: Callable = _ensemble_stats_aggregation_fun,
    save_path: str | None = None,
    label: str | None = None,
    overwrite: bool = True,
) -> xr.DataArray | None:
    """Compute ensemble statistics using a specified function.

    Args:
        data (xarray.Dataset): Input ensemble data
        indicator_fun: Function to compute the climatic indicator
        indicator_kwargs: Keyword arguments for indicator function
        aggregation_fun: Function to aggregate ensemble (default: mean, variance, count)
        save_path: Directory to save results
        label: Label to add to filename
        overwrite: Whether to overwrite existing files
    """
    kwargs = indicator_kwargs or {}
    print(f"    Computing indicator: {getattr(indicator_fun, '__name__', 'indicator')}")

    # Prepare save path if provided
    if save_path:
        save_path = os.path.expanduser(save_path)
        os.makedirs(save_path, exist_ok=True)

        # Check if files already exist before computing
        if not overwrite and _check_files_exist(indicator_fun, label, save_path):
            print("      Found existing files matching pattern. Skipping computation.")
            return None

    # Compute indicator and aggregate ensemble statistics
    required_vars = get_required_variables(indicator_fun)

    # Get function signature to preserve parameter order
    sig = inspect.signature(indicator_fun)
    param_names = [
        name
        for name in sig.parameters.keys()
        if name in required_vars and sig.parameters[name].default is inspect.Parameter.empty
    ]

    args = [data[var] for var in param_names]
    results = indicator_fun(*args, **kwargs).compute()

    results = aggregation_fun(results).compute()
    if not isinstance(results, xr.DataArray):
        raise TypeError("Aggregation function must return an xarray.DataArray")

    # Save results if path is provided
    if save_path:
        indicator_name = getattr(indicator_fun, "__name__", "indicator")

        if "return_period" in results.dims:
            # Handle return period data by splitting into separate files
            # (All return period indicators end with "_rp")
            return_periods = results.coords["return_period"].values
            print(f"    Found return periods: {return_periods}")

            saved_files = []
            for rp in return_periods:
                filename = f"{indicator_name}{int(rp)}"  # "rx1day_rp2"
                if label:
                    filename = f"{filename}_{label}"
                filepath = os.path.join(save_path, filename + ".nc")

                rp_data = results.sel(return_period=rp)
                if _save_data_array(rp_data, filepath, overwrite):
                    saved_files.append(filepath)

            print(f"    Saved {len(saved_files)} return period files")
        else:
            # Standard single file case
            base_filename = _build_filename(indicator_fun, label)
            filepath = os.path.join(save_path, base_filename + ".nc")
            _save_data_array(results, filepath, overwrite)

    return results
