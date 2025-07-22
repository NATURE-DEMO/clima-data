"""Pytest configuration and shared fixtures."""

import tempfile

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def sample_xarray_dataset():
    """Create sample xarray dataset for testing."""
    # Create coordinate arrays
    x = np.linspace(-30, 20, 50)
    y = np.linspace(-25, 25, 50)

    # Create sample data
    rx1day_data = np.random.exponential(10, (50, 50)) + 5

    # Create dataset
    ds = xr.Dataset(
        {"rx1day": (["y", "x"], rx1day_data)},
        coords={"x": x, "y": y},
        attrs={"title": "Test Climate Data"},
    )

    # Add attributes to data variable
    ds["rx1day"].attrs = {"units": "mm", "long_name": "Maximum 1-day precipitation"}

    return ds


@pytest.fixture
def sample_city_data():
    """Sample city search results."""
    return [
        {
            "name": "Berlin, Deutschland",
            "lat": 52.5200,
            "lon": 13.4050,
            "boundingbox": ["52.3671", "52.6755", "13.0884", "13.7611"],
            "geojson": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [13.0884, 52.3671],
                        [13.7611, 52.3671],
                        [13.7611, 52.6755],
                        [13.0884, 52.6755],
                        [13.0884, 52.3671],
                    ]
                ],
            },
            "osm_id": "62422",
            "type": "city",
        },
        {
            "name": "Paris, France",
            "lat": 48.8566,
            "lon": 2.3522,
            "boundingbox": ["48.8155", "48.9021", "2.2241", "2.4699"],
            "type": "city",
        },
    ]


@pytest.fixture
def sample_area_city():
    """Sample area data for city."""
    return {
        "name": "Berlin, Deutschland",
        "type": "city",
        "lat": 52.5200,
        "lon": 13.4050,
        "boundingbox": ["52.3671", "52.6755", "13.0884", "13.7611"],
    }


@pytest.fixture
def sample_area_rectangle():
    """Sample area data for rectangle."""
    return {
        "name": "Custom Rectangle Area",
        "type": "rectangle",
        "bounds": {"north": 55.0, "south": 50.0, "east": 15.0, "west": 10.0},
    }


@pytest.fixture
def temp_data_directory():
    """Create temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
