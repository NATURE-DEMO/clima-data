"""
Test realization alignment functionality across variables.
Tests for the new alignment system to ensure same realization indices
represent the same climate models across different variables.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from clima_data.utils import extract_model_metadata, validate_realization_alignment


def test_extract_model_metadata():
    """Test model metadata extraction with various CORDEX filename formats."""
    test_files = [
        "tasmax_EUR-11_ICHEC-EC-EARTH_rcp85_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1_v1_day_20410101-20451231.nc",
        "pr_EUR-11_MPI-M-MPI-ESM-LR_rcp85_r1i1p1_SMHI-RCA4_v1a_day_20810101-20851231.nc",
        "tas_EUR-11_CNRM-CERFACS-CNRM-CM5_historical_r1i1p1_GERICS-REMO2015_v2_mon_197101-198012.nc",
    ]

    expected_model_ids = [
        "ICHEC-EC-EARTH_r1i1p1_CLMcom-ETH-COSMO-crCLIM-v1-1",
        "MPI-M-MPI-ESM-LR_r1i1p1_SMHI-RCA4",
        "CNRM-CERFACS-CNRM-CM5_r1i1p1_GERICS-REMO2015",
    ]

    for filename, expected_id in zip(test_files, expected_model_ids, strict=False):
        metadata = extract_model_metadata(filename)

        assert metadata["model_id"] == expected_id
        assert "gcm" in metadata
        assert "rcm" in metadata
        assert "ensemble" in metadata
        assert "experiment" in metadata


def test_extract_model_metadata_invalid_format():
    """Test that invalid filenames raise appropriate errors."""
    invalid_files = [
        "pr_EUR-11_ICHEC-EC-EARTH_rcp85_r1i1p1.nc",  # Missing components
        "not_a_cordex_file.nc",  # Wrong format
        "pr_EUR-11_ICHEC-EC-EARTH.nc",  # Too few components
    ]

    for filename in invalid_files:
        with pytest.raises(ValueError, match="Invalid CORDEX filename format"):
            extract_model_metadata(filename)


def test_validate_realization_alignment_success():
    """Test realization alignment validation with properly aligned datasets."""
    # Create mock datasets with matching realization coordinates
    time = pd.date_range("2000-01-01", periods=10, freq="D")
    x = np.arange(5)
    y = np.arange(3)
    realizations = ["MODEL1_r1i1p1_RCM1", "MODEL2_r1i1p1_RCM2"]

    # Dataset 1 (e.g., precipitation)
    data1 = np.random.random((len(realizations), len(time), len(y), len(x)))
    ds1 = xr.Dataset(
        {"pr": (["realization", "time", "y", "x"], data1)},
        coords={"realization": realizations, "time": time, "y": y, "x": x},
    )

    # Dataset 2 (e.g., temperature) - same realization coordinate
    data2 = np.random.random((len(realizations), len(time), len(y), len(x)))
    ds2 = xr.Dataset(
        {"tas": (["realization", "time", "y", "x"], data2)},
        coords={"realization": realizations, "time": time, "y": y, "x": x},
    )

    # Should not raise any exception
    validate_realization_alignment([ds1, ds2])


def test_validate_realization_alignment_misaligned():
    """Test that misaligned realization coordinates raise ValueError."""
    time = pd.date_range("2000-01-01", periods=10, freq="D")
    x = np.arange(5)
    y = np.arange(3)

    # Dataset 1 with one set of realizations
    realizations1 = ["MODEL1_r1i1p1_RCM1", "MODEL2_r1i1p1_RCM2"]
    data1 = np.random.random((len(realizations1), len(time), len(y), len(x)))
    ds1 = xr.Dataset(
        {"pr": (["realization", "time", "y", "x"], data1)},
        coords={"realization": realizations1, "time": time, "y": y, "x": x},
    )

    # Dataset 2 with different realizations
    realizations2 = ["MODEL1_r1i1p1_RCM1", "MODEL3_r1i1p1_RCM3"]  # Different second model
    data2 = np.random.random((len(realizations2), len(time), len(y), len(x)))
    ds2 = xr.Dataset(
        {"tas": (["realization", "time", "y", "x"], data2)},
        coords={"realization": realizations2, "time": time, "y": y, "x": x},
    )

    with pytest.raises(ValueError, match="realization coordinate differs"):
        validate_realization_alignment([ds1, ds2])


def test_validate_realization_alignment_different_sizes():
    """Test that different realization sizes raise ValueError."""
    time = pd.date_range("2000-01-01", periods=10, freq="D")
    x = np.arange(5)
    y = np.arange(3)

    # Dataset 1 with 2 realizations
    realizations1 = ["MODEL1_r1i1p1_RCM1", "MODEL2_r1i1p1_RCM2"]
    data1 = np.random.random((len(realizations1), len(time), len(y), len(x)))
    ds1 = xr.Dataset(
        {"pr": (["realization", "time", "y", "x"], data1)},
        coords={"realization": realizations1, "time": time, "y": y, "x": x},
    )

    # Dataset 2 with 3 realizations
    realizations2 = ["MODEL1_r1i1p1_RCM1", "MODEL2_r1i1p1_RCM2", "MODEL3_r1i1p1_RCM3"]
    data2 = np.random.random((len(realizations2), len(time), len(y), len(x)))
    ds2 = xr.Dataset(
        {"tas": (["realization", "time", "y", "x"], data2)},
        coords={"realization": realizations2, "time": time, "y": y, "x": x},
    )

    with pytest.raises(ValueError, match="has 3 realizations, expected 2"):
        validate_realization_alignment([ds1, ds2])


def test_validate_realization_alignment_missing_dimension():
    """Test that missing realization dimension raises ValueError."""
    time = pd.date_range("2000-01-01", periods=10, freq="D")
    x = np.arange(5)
    y = np.arange(3)

    # Dataset 1 with realization dimension
    realizations = ["MODEL1_r1i1p1_RCM1", "MODEL2_r1i1p1_RCM2"]
    data1 = np.random.random((len(realizations), len(time), len(y), len(x)))
    ds1 = xr.Dataset(
        {"pr": (["realization", "time", "y", "x"], data1)},
        coords={"realization": realizations, "time": time, "y": y, "x": x},
    )

    # Dataset 2 without realization dimension
    data2 = np.random.random((len(time), len(y), len(x)))
    ds2 = xr.Dataset({"tas": (["time", "y", "x"], data2)}, coords={"time": time, "y": y, "x": x})

    with pytest.raises(ValueError, match="Dataset 1 missing realization dimension"):
        validate_realization_alignment([ds1, ds2])


def test_validate_realization_alignment_single_dataset():
    """Test that validation passes for single dataset (no validation needed)."""
    time = pd.date_range("2000-01-01", periods=10, freq="D")
    x = np.arange(5)
    y = np.arange(3)
    realizations = ["MODEL1_r1i1p1_RCM1"]

    data = np.random.random((len(realizations), len(time), len(y), len(x)))
    ds = xr.Dataset(
        {"pr": (["realization", "time", "y", "x"], data)},
        coords={"realization": realizations, "time": time, "y": y, "x": x},
    )

    # Should not raise any exception
    validate_realization_alignment([ds])
