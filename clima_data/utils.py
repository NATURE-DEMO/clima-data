import inspect
import os
import re
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import xarray as xr

from clima_data.cordex import get_files

# Recognized climate variable names for automatic detection in function signatures
RECOGNIZED_CLIMATE_VARIABLES = {
    "pr",        # precipitation
    "tas",       # near-surface air temperature
    "tasmin",    # daily minimum near-surface air temperature
    "tasmax",    # daily maximum near-surface air temperature
    "hurs",      # near-surface relative humidity
    "sfcWind",   # near-surface wind speed
    "ps",        # surface air pressure
    "rsds",      # surface downwelling shortwave radiation
}


def get_cds_key() -> str:
    key = os.getenv("CDS_KEY")
    if not key:
        raise RuntimeError("CDS_KEY not set.")
    return key


def group_filenames_by_base(file_list: list[str]) -> dict[str, list[str]]:
    # Match either _YYYYMMDD-YYYYMMDD.nc or _YYYYMM-YYYYMM.nc at the end of the file.
    date_pattern = re.compile(r"_(\d{8}-\d{8}|\d{6}-\d{6})\.nc$")

    groups = defaultdict(list)

    for f in file_list:
        # Extract just the filename from the full path
        file_name = f.split("/")[-1]
        # Remove the date part if it exists.
        base_name = date_pattern.sub("", file_name)
        groups[base_name].append(f)

    # Sort file names within each group
    for base_name in groups:
        groups[base_name].sort()

    return dict(groups)


def select_years_within_frame(
    groups: list[list[str]], year_start: int, year_end: int
) -> list[list[str]]:
    """Select only files whose years are within the specified range.

    Args:
        groups: Dictionary of grouped filenames
        year_start: Start year of the range (inclusive)
        year_end: End year of the range (inclusive)

    Example:
        >>> groups = [
        ...     [
        ...         "file_19800101-19801231.nc",
        ...         "file_19810101-19811231.nc",
        ...         "file_19850101-19951231.nc",
        ...     ],
        ...     [
        ...         "file_19800101-19901231.nc",
        ...         "file_1990001-200012.nc",
        ...         "file_200101-200112.nc",
        ...     ],
        ... ]
        >>> filtered_groups = select_years_within_frame(groups, year_start=1980, year_end=1990)
        >>> len(filtered_groups)
        2
    """
    filtered_groups = []
    for group in groups:
        filtered_group = []
        for filename in group:
            # Extract year_start and year_end from _YYYYMMDD-YYYYMMDD.nc or _YYYYMM-YYYYMM.nc at the end of the file:
            match = re.search(r"(\d{4})\d*-(\d{4})\d*\.nc$", filename)
            if match:
                y1, y2 = int(match.group(1)), int(match.group(2))
                if year_start <= y1 and y2 <= year_end:
                    filtered_group.append(filename)
        filtered_groups.append(filtered_group)
    return filtered_groups


def align_coordinates(
    arrays: list[xr.DataArray],
    reference_index: int = 0,
    exclude_dims: set[str] | None = None,
    tolerance: float = 1e-4,
) -> list[xr.DataArray]:
    """Align coordinates using reference array, excluding specified dimensions.

    Args:
        arrays: DataArrays to align
        reference_index: Index of reference array (default: 0)
        exclude_dims: Coordinate names to exclude from alignment
        tolerance: Tolerance for numeric coordinate comparison

    Returns:
        List of aligned DataArrays

    Raises:
        ValueError: If coordinates differ significantly

    Example:
        >>> import xarray as xr
        >>> import numpy as np
        >>> da1 = xr.DataArray([1, 2, 3], coords={'x': [1, 2, 3]})
        >>> da2 = xr.DataArray([4, 5, 6], coords={'x': [1.0001, 2.0001, 3.0001]})
        >>> arrays = [da1, da2]
        >>> aligned = align_coordinates(arrays)
        >>> len(aligned)
        2
    """
    exclude_dims = exclude_dims or set()
    ref_coords = {
        name: coord
        for name, coord in arrays[reference_index].coords.items()
        if name not in exclude_dims
    }

    # Validate coordinate compatibility
    for i, arr in enumerate(arrays):
        if i == reference_index:
            continue

        for name, ref_coord in ref_coords.items():
            if name not in arr.coords:
                raise ValueError("Missing coordinate.")

            arr_coord = arr.coords[name]
            if ref_coord.shape != arr_coord.shape:
                raise ValueError("Coordinate shape mismatch.")

            # Check numeric coordinate values
            if np.issubdtype(ref_coord.dtype, np.number):
                if not np.allclose(
                    ref_coord.values,
                    arr_coord.values,
                    rtol=tolerance,
                    atol=tolerance,
                    equal_nan=True,
                ):
                    raise ValueError("Coordinate values differ.")
            elif not ref_coord.equals(arr_coord):
                raise ValueError("Coordinate values differ.")

    return [
        arr if i == reference_index else arr.assign_coords(ref_coords)
        for i, arr in enumerate(arrays)
    ]


def extract_model_metadata(filename: str) -> dict[str, str]:
    """Extract model metadata from CORDEX filename.

    Args:
        filename: CORDEX filename following pattern:
            <variable>_EUR-11_<gcm>_<experiment>_<ensemble>_<rcm>_<version>_<freq>_<dates>.nc

    Returns:
        Dictionary with model metadata

    Example:
        >>> metadata = extract_model_metadata("pr_EUR-11_CNRM-CERFACS-CNRM-CM5_rcp85_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_day_20060101-20101231.nc")
        >>> metadata['rcm']
        'CLMcom-ETH-COSMO-crCLIM-v1-1'
    """
    basename = os.path.basename(filename)
    # Remove .nc extension and date part
    basename = re.sub(r"_\d{8}-\d{8}\.nc$", "", basename)

    # Split by underscore to extract components
    parts = basename.split("_")

    if len(parts) < 7:
        raise ValueError("Invalid CORDEX filename format.")

    return {
        "variable": parts[0],
        "domain": parts[1],
        "gcm": parts[2],
        "experiment": parts[3],
        "ensemble": parts[4],
        "rcm": parts[5],
        "version": parts[6],
        "model_id": f"{parts[2]}_{parts[4]}_{parts[5]}",  # unique model identifier
    }


def get_common_file_groups(
    variables: list[str],
    cordex_path: str,
    experiment: str,
    frequency: str,
    year_start: int,
    year_end: int,
) -> tuple[list[list[list[str]]], list[dict[str, str]]]:
    """Get file groups that exist for ALL specified variables with consistent model ordering.

    Args:
        variables: List of climate variables to process
        cordex_path: Path to CORDEX data directory
        experiment: Climate experiment (e.g., "historical", "rcp45")
        frequency: Data frequency (e.g., "day")
        year_start: Start year for time selection
        year_end: End year for time selection

    Returns:
        Tuple of (file_groups_per_variable, model_metadata_list)
        - file_groups_per_variable: List of (list of file groups), one per variable
        - model_metadata_list: List of model metadata dicts for each realization

    Raises:
        ValueError: If no common models found across all variables
    """
    # Get file groups for each variable
    variable_file_groups = {}
    variable_models = {}

    for variable in variables:
        file_list = get_files(cordex_path, variable, experiment, frequency)
        file_groups_dict = group_filenames_by_base(file_list)
        file_groups_list = list(file_groups_dict.values())
        file_groups_filtered = select_years_within_frame(file_groups_list, year_start, year_end)

        # Extract model metadata for each group
        models_for_variable = []
        for group in file_groups_filtered:
            if group:  # Skip empty groups
                metadata = extract_model_metadata(group[0])  # Use first file as representative
                models_for_variable.append(metadata)

        variable_file_groups[variable] = file_groups_filtered
        variable_models[variable] = models_for_variable

    # Find common models across all variables
    if not variables:
        return [], []

    # Start with models from first variable
    common_model_ids = {model["model_id"] for model in variable_models[variables[0]]}

    # Intersect with models from other variables
    for variable in variables[1:]:
        variable_model_ids = {model["model_id"] for model in variable_models[variable]}
        common_model_ids &= variable_model_ids

    if not common_model_ids:
        raise ValueError("No common models found across variables.")

    # Create consistent ordering based on model_id sorting
    common_models_sorted = sorted(common_model_ids)

    # Build aligned file groups and metadata
    aligned_file_groups = []
    model_metadata_list = []

    for variable in variables:
        variable_groups = []
        for model_id in common_models_sorted:
            # Find the file group for this model_id
            for i, model_meta in enumerate(variable_models[variable]):
                if model_meta["model_id"] == model_id:
                    variable_groups.append(variable_file_groups[variable][i])
                    # Store metadata only once (from first variable)
                    if variable == variables[0]:
                        model_metadata_list.append(model_meta)
                    break
        aligned_file_groups.append(variable_groups)

    return aligned_file_groups, model_metadata_list


def validate_realization_alignment(datasets: list[xr.Dataset]) -> None:
    """Validate that realization coordinates represent the same models across datasets.

    Args:
        datasets: List of datasets to validate

    Raises:
        ValueError: If realization alignment is inconsistent
    """
    if len(datasets) <= 1:
        return

    # Check that all datasets have realization dimension
    for i, ds in enumerate(datasets):
        if "realization" not in ds.dims:
            raise ValueError(f"Dataset {i} missing realization dimension")

    # Check realization dimension sizes match
    ref_size = datasets[0].sizes["realization"]
    for i, ds in enumerate(datasets[1:], 1):
        if ds.sizes["realization"] != ref_size:
            raise ValueError(
                f"Dataset {i} has {ds.sizes['realization']} realizations, expected {ref_size}"
            )

    # If realization coordinate has model metadata, validate it matches
    ref_ds = datasets[0]
    if "realization" in ref_ds.coords:
        ref_coord = ref_ds.coords["realization"]

        for i, ds in enumerate(datasets[1:], 1):
            if "realization" not in ds.coords:
                continue

            ds_coord = ds.coords["realization"]

            # Check coordinate values match
            if not ref_coord.equals(ds_coord):
                raise ValueError(
                    f"Dataset {i} realization coordinate differs from reference. This indicates model misalignment across variables."
                )

    print(f"✓ Realization alignment validated across {len(datasets)} datasets")


def get_required_variables(func: Callable) -> set[str]:
    """Extract required climate variables from function signature."""
    sig = inspect.signature(func)
    variables = set()

    for param_name, param in sig.parameters.items():
        # Skip optional parameters like cal_start, cal_end
        if param.default is not inspect.Parameter.empty:
            continue
        # Add known climate variables
        if param_name in RECOGNIZED_CLIMATE_VARIABLES:
            variables.add(param_name)

    return variables
