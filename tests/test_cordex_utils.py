import cartopy.crs as ccrs
import pyproj
import pytest

from clima_data import CordexNames

WGS84_TO_CORDEX_TRANS = CordexNames.WGS84_TO_CORDEX_TRANS


def test_coordinate_transformation():
    # WGS84 bounds of Catalonia
    west, south, east, north = [0, 40.2, 3.3, 43.5]

    # Transform corner points individually to get bounds
    # Note: WGS84_TO_CORDEX_TRANS expects (latitude, longitude) order
    sw = WGS84_TO_CORDEX_TRANS.transform(south, west)
    se = WGS84_TO_CORDEX_TRANS.transform(south, east)
    nw = WGS84_TO_CORDEX_TRANS.transform(north, west)
    ne = WGS84_TO_CORDEX_TRANS.transform(north, east)

    xs = [sw[0], se[0], nw[0], ne[0]]
    ys = [sw[1], se[1], nw[1], ne[1]]

    RLON_MIN, RLON_MAX = min(xs), max(xs)
    RLAT_MIN, RLAT_MAX = min(ys), max(ys)

    # Ground truth values for the transformation of Catalonia coordinates:
    RLON_MIN_cordex = -13.82050453871809
    RLAT_MIN_cordex = -9.423645743863924
    RLON_MAX_cordex = -10.66980176143207
    RLAT_MAX_cordex = -5.66467325395815

    assert RLON_MIN_cordex == pytest.approx(RLON_MIN, abs=1e-6)
    assert RLAT_MIN_cordex == pytest.approx(RLAT_MIN, abs=1e-6)
    assert RLON_MAX_cordex == pytest.approx(RLON_MAX, abs=1e-6)
    assert RLAT_MAX_cordex == pytest.approx(RLAT_MAX, abs=1e-6)

    # Test with individual point transformation for consistency
    cordex_crs = ccrs.RotatedPole(pole_latitude=39.25, pole_longitude=-162)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", cordex_crs, always_xy=True)

    # Transform corner points with the alternative transformer
    sw_alt = transformer.transform(west, south)

    # Should get similar results (within reasonable tolerance for different implementations)
    assert abs(sw[0] - sw_alt[0]) < 0.1
    assert abs(sw[1] - sw_alt[1]) < 0.1
