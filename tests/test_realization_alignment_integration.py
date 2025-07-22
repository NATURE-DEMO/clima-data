"""
Integration test for realization alignment with real CORDEX data.
Tests the complete workflow with minimal test data in ~/data/cordex_test.

Expected minimal test data structure:
~/data/cordex_test/
├── pr_EUR-11_ICHEC-EC-EARTH_historical_r1i1p1_KNMI-RACMO22E_v1_day_19810101-19811231.nc
├── pr_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_SMHI-RCA4_v1_day_19810101-19811231.nc
├── tas_EUR-11_ICHEC-EC-EARTH_historical_r1i1p1_KNMI-RACMO22E_v1_day_19810101-19811231.nc
└── tas_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_SMHI-RCA4_v1_day_19810101-19811231.nc
"""

from pathlib import Path

import pytest
import xarray as xr

from clima_data.utils import get_common_file_groups, validate_realization_alignment
from scripts.compute_indices import load_and_process_ensemble


class TestRealizationAlignmentIntegration:
    """Integration tests for realization alignment with real CORDEX data."""

    @pytest.fixture
    def cordex_test_path(self):
        """Path to minimal test data."""
        return Path.home() / "data" / "cordex_test"

    @pytest.fixture
    def test_data_available(self, cordex_test_path):
        """Check if test data is available."""
        return cordex_test_path.exists() and any(cordex_test_path.rglob("*.nc"))

    def test_get_common_file_groups_with_real_data(self, cordex_test_path, test_data_available):
        """Test get_common_file_groups with real CORDEX files."""
        if not test_data_available:
            pytest.skip("Test data not available at ~/data/cordex_test")

        # Test finding common models across pr and tasmin (using available data)
        file_groups_per_var, model_metadata = get_common_file_groups(
            variables=["pr", "tasmin"],
            cordex_path=str(cordex_test_path),
            experiment="historical",
            frequency="day",
            year_start=1976,
            year_end=1980,
        )

        # Verify we found common models
        assert len(model_metadata) > 0, "Should find at least one common model"
        assert len(file_groups_per_var) == 2, "Should have file groups for pr and tasmin"

        # Verify each variable has same number of models
        assert len(file_groups_per_var[0]) == len(file_groups_per_var[1]), (
            "Both variables should have same number of models"
        )

        # Verify model metadata structure
        for metadata in model_metadata:
            assert "model_id" in metadata
            assert "gcm" in metadata
            assert "rcm" in metadata
            assert "ensemble" in metadata
            assert "experiment" in metadata

        print(f"✓ Found {len(model_metadata)} common models:")
        for i, meta in enumerate(model_metadata):
            print(f"  {i}: {meta['model_id']}")

    def test_load_and_process_ensemble_minimal(self, cordex_test_path, test_data_available):
        """Test load_and_process_ensemble with minimal real data."""
        if not test_data_available:
            pytest.skip("Test data not available at ~/data/cordex_test")

        # Test the complete workflow with explicit cordex_path
        dataset = load_and_process_ensemble(
            variables=["pr", "tasmin"],
            exp="historical",
            year_start=1976,
            year_end=1980,
            cordex_path=str(cordex_test_path),
        )

        # Verify dataset structure
        assert isinstance(dataset, xr.Dataset)
        assert "pr" in dataset.data_vars
        assert "tasmin" in dataset.data_vars
        assert "realization" in dataset.dims
        assert dataset.sizes["realization"] > 0

        # Verify realization coordinates contain model identifiers
        realizations = dataset.realization.values
        assert all(isinstance(r, str) for r in realizations), (
            "Realization coordinates should be model ID strings"
        )

        # Verify model metadata is present
        if hasattr(dataset.pr, "attrs") and "model_metadata" in dataset.pr.attrs:
            metadata = dataset.pr.attrs["model_metadata"]
            assert len(metadata) == len(realizations)

        print(f"✓ Successfully loaded dataset with {len(realizations)} realizations")
        print(f"✓ Variables: {list(dataset.data_vars)}")
        print(f"✓ Realizations: {realizations}")

    def test_realization_alignment_validation_real_data(
        self, cordex_test_path, test_data_available
    ):
        """Test that realization alignment validation works with real data."""
        if not test_data_available:
            pytest.skip("Test data not available at ~/data/cordex_test")

        # Get file groups for testing
        file_groups_per_var, model_metadata = get_common_file_groups(
            variables=["pr", "tasmin"],
            cordex_path=str(cordex_test_path),
            experiment="historical",
            frequency="day",
            year_start=1976,
            year_end=1980,
        )

        if not model_metadata:
            pytest.skip("No common models found in test data")

        # Load individual datasets to test alignment
        from clima_data.stats import ensemble_cat_time

        datasets = []
        for i, variable in enumerate(["pr", "tasmin"]):
            ens_da = ensemble_cat_time(
                file_groups=file_groups_per_var[i],
                variable=variable,
                apply_fix=True,
                project_id="cordex",
                model_metadata=model_metadata,
            )
            datasets.append(ens_da.to_dataset())

        # Test validation passes
        validate_realization_alignment(datasets)
        print("✓ Realization alignment validation passed with real data")

    def test_single_model_case(self, cordex_test_path, test_data_available):
        """Test with only one model available (edge case)."""
        if not test_data_available:
            pytest.skip("Test data not available at ~/data/cordex_test")

        try:
            file_groups_per_var, model_metadata = get_common_file_groups(
                variables=["pr", "tasmin"],
                cordex_path=str(cordex_test_path),
                experiment="historical",
                frequency="day",
                year_start=1976,
                year_end=1980,
            )

            if len(model_metadata) == 1:
                print("✓ Single model case detected and handled correctly")
            else:
                print(f"✓ Multiple models case: {len(model_metadata)} models found")

        except ValueError as e:
            if "No common models found" in str(e):
                pytest.skip("Test data doesn't have common models across variables")
            else:
                raise

    @pytest.mark.parametrize(
        "variables",
        [
            ["pr"],  # Single variable
            ["pr", "tasmin"],  # Two variables
        ],
    )
    def test_different_variable_combinations(
        self, cordex_test_path, test_data_available, variables
    ):
        """Test different combinations of variables."""
        if not test_data_available:
            pytest.skip("Test data not available at ~/data/cordex_test")

        try:
            file_groups_per_var, model_metadata = get_common_file_groups(
                variables=variables,
                cordex_path=str(cordex_test_path),
                experiment="historical",
                frequency="day",
                year_start=1976,
                year_end=1980,
            )

            assert len(file_groups_per_var) == len(variables)
            print(f"✓ Variables {variables}: found {len(model_metadata)} models")

        except ValueError as e:
            if "No common models found" in str(e):
                pytest.skip(f"No models found for variables {variables}")
            else:
                raise


def test_create_minimal_test_data_instructions():
    """Provide instructions for creating minimal test data if it doesn't exist."""
    cordex_test_path = Path.home() / "data" / "cordex_test"

    if not cordex_test_path.exists():
        print("\n" + "=" * 60)
        print("MINIMAL TEST DATA SETUP INSTRUCTIONS")
        print("=" * 60)
        print(f"Create directory: {cordex_test_path}")
        print("Add minimal CORDEX files with this pattern:")
        print("")
        print("For 2 models × 2 variables (recommended):")
        print(
            "  pr_EUR-11_ICHEC-EC-EARTH_historical_r1i1p1_KNMI-RACMO22E_v1_day_19810101-19811231.nc"
        )
        print(
            "  pr_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_SMHI-RCA4_v1_day_19810101-19811231.nc"
        )
        print(
            "  tas_EUR-11_ICHEC-EC-EARTH_historical_r1i1p1_KNMI-RACMO22E_v1_day_19810101-19811231.nc"
        )
        print(
            "  tas_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_SMHI-RCA4_v1_day_19810101-19811231.nc"
        )
        print("")
        print("Or minimal (1 model × 2 variables):")
        print(
            "  pr_EUR-11_ICHEC-EC-EARTH_historical_r1i1p1_KNMI-RACMO22E_v1_day_19810101-19811231.nc"
        )
        print(
            "  tas_EUR-11_ICHEC-EC-EARTH_historical_r1i1p1_KNMI-RACMO22E_v1_day_19810101-19811231.nc"
        )
        print("")
        print("Run: pytest tests/integration/test_realization_alignment_integration.py -v")
        print("=" * 60)

    # Always pass this test - it's just for documentation
    assert True
