import xarray as xr
import xclim.indicators.atmos as xa
from xclim.indices.stats import frequency_analysis

## Precipitation climatic indices


def prcptot_year(pr: xr.DataArray) -> xr.DataArray:
    """**Total yearly precipitation (solid and liquid)**

    Total yearly accumulated precipitation (solid and liquid).

    | Metadata      | Value                                   |
    |-------------- |-----------------------------------------|
    | Identifier    | prcptot                                 |
    | Units         | mm year-1                               |
    | Frequency     | YS                                      |
    | Standard Name | lwe_thickness_of_precipitation_amount   |

    Args:
        pr: Precipitation (as an xarray DataArray).

    Returns:
        xarray.DataArray: Total yearly accumulated precipitation.
    """
    return xa.precip_accumulation(pr=pr, freq="YS")  # type: ignore


def rx1day(pr: xr.DataArray) -> xr.DataArray:
    """**Maximum 1-day precipitation amount**

    Maximum precipitation amount over a single day.

    | Metadata      | Value                                   |
    |-------------- |-----------------------------------------|
    | Identifier    | rx1day                                  |
    | Units         | mm day-1                                |
    | Frequency     | YS                                      |
    | Standard Name | maximum_1_day_precipitation_amount      |

    Args:
        pr: Precipitation (as an xarray DataArray).

    Returns:
        xarray.DataArray: Maximum 1-day precipitation amount.
    """
    return xa.max_1day_precipitation_amount(pr, freq="YS")  # type: ignore


def rx5day(pr: xr.DataArray) -> xr.DataArray:
    """**Maximum 5-day precipitation amount**

    Maximum precipitation amount over a single day.

    | Metadata      | Value                                   |
    |-------------- |-----------------------------------------|
    | Identifier    | rx5day                                  |
    | Units         | mm day-1                                |
    | Frequency     | YS                                      |
    | Standard Name | maximum_5_day_precipitation_amount      |

    Args:
        pr: Precipitation (as an xarray DataArray).

    Returns:
        xarray.DataArray: Maximum 5-day precipitation amount.
    """
    return xa.max_n_day_precipitation_amount(pr, window=5, freq="YS")  # type: ignore


def rx1day_rp(pr: xr.DataArray) -> xr.DataArray:
    """**100-year return level of rx1day**

    Return level of maximum 1-day precipitation for periods of 2 to 100 years.

    | Metadata      | Value                                                   |
    |-------------- |---------------------------------------------------------|
    | Identifier    | rx1day_rp                                               |
    | Units         | mm day-1                                                |
    | Standard Name | maximum_1_day_precipitation_amount_X_year_return_period |

    Args:
        pr: Precipitation (as an xarray DataArray).

    Returns:
        Computed return levels as an xarray.DataArray.
    """
    rx1day_ = rx1day(pr)
    results = frequency_analysis(
        rx1day_,
        mode="max",
        t=[2, 5, 10, 25, 50, 100, 200, 500],
        dist="genextreme",
        freq="YS",
        method="ML",
    )
    return results  # type: ignore


def rx5day_rp(pr: xr.DataArray) -> xr.DataArray:
    """**100-year return level of rx1day**

    Return level of maximum 5-day precipitation for periods of 2 to 100 years.

    | Metadata      | Value                                                   |
    |-------------- |---------------------------------------------------------|
    | Identifier    | rx5day_rp                                               |
    | Units         | mm day-1                                                |
    | Standard Name | maximum_1_day_precipitation_amount_X_year_return_period |

    Args:
        pr: Precipitation (as an xarray DataArray).

    Returns:
        Computed return levels as an xarray.DataArray.
    """
    rx5day_ = rx5day(pr)
    results = frequency_analysis(
        rx5day_,
        mode="max",
        t=[2, 5, 10, 25, 50, 100, 200, 500],
        dist="genextreme",
        freq="YS",
        method="ML",
    )
    return results  # type: ignore


def cwd(pr: xr.DataArray) -> xr.DataArray:
    """**Maximum consecutive wet days**

    Maximum number of consecutive days with precipitation above 1 mm/day.

    | Metadata      | Value                                                           |
    |-------------- |-----------------------------------------------------------------|
    | Identifier    | cwd                                                             |
    | Units         | days                                                            |
    | Frequency     | YS                                                              |
    | Standard Name | number_of_days_with_lwe_thickness_of_precipitation_amount_at_or_above_threshold |

    Args:
        pr: Precipitation (as an xarray DataArray).

    Returns:
        xarray.DataArray: Maximum consecutive wet days.
    """
    return xa.maximum_consecutive_wet_days(pr, thresh="1 mm/day", freq="YS")  # type: ignore


def cdd(pr: xr.DataArray) -> xr.DataArray:
    """**Maximum consecutive dry days**

    Maximum number of consecutive days with precipitation below 1 mm/day.

    | Metadata      | Value                                                           |
    |-------------- |-----------------------------------------------------------------|
    | Identifier    | cdd                                                             |
    | Units         | days                                                            |
    | Frequency     | YS                                                              |
    | Standard Name | number_of_days_with_lwe_thickness_of_precipitation_amount_below_threshold |

    Args:
        pr: Precipitation (as an xarray DataArray).

    Returns:
        xarray.DataArray: Maximum consecutive dry days.
    """
    return xa.maximum_consecutive_dry_days(pr, thresh="1 mm/day", freq="YS")  # type: ignore


"""Temperature climatic indices"""


def tg_year(tas: xr.DataArray) -> xr.DataArray:
    """**Annual mean average temperature**

    Mean of daily mean temperature aggregated yearly.

    | Metadata      | Value           |
    |-------------- |-----------------|
    | Identifier    | tg              |
    | Units         | K               |
    | Frequency     | YS              |
    | Standard Name | air_temperature |

    Args:
        tas: Air temperature (as an xarray DataArray).

    Returns:
        xarray.DataArray: Annual mean temperature.
    """
    return xa.tg_mean(tas=tas, freq="YS")  # type: ignore


def tn_year(tasmin: xr.DataArray) -> xr.DataArray:
    """**Annual mean minimum temperature**

    Mean of daily minimum temperature aggregated yearly.

    | Metadata      | Value           |
    |-------------- |-----------------|
    | Identifier    | tn              |
    | Units         | K               |
    | Frequency     | YS              |
    | Standard Name | air_temperature |

    Args:
        tasmin: Daily minimum temperature (as an xarray DataArray).

    Returns:
        xarray.DataArray: Annual mean minimum temperature.
    """
    return xa.tn_mean(tasmin=tasmin, freq="YS")  # type: ignore


def tx_year(tasmax: xr.DataArray) -> xr.DataArray:
    """**Annual mean maximum temperature**

    Mean of daily maximum temperature aggregated yearly.

    | Metadata      | Value           |
    |-------------- |-----------------|
    | Identifier    | tx              |
    | Units         | K               |
    | Frequency     | YS              |
    | Standard Name | air_temperature |

    Args:
        tasmax: Daily maximum temperature (as an xarray DataArray).

    Returns:
        xarray.DataArray: Annual mean maximum temperature.
    """
    return xa.tx_mean(tasmax=tasmax, freq="YS")  # type: ignore


def tx40(tasmax: xr.DataArray) -> xr.DataArray:
    """**Annual days with maximum temperature > 40°C**

    Average number of days per year with daily maximum temperature > 40°C.

    | Metadata      | Value                                         |
    |-------------- |-----------------------------------------------|
    | Identifier    | tx40                                          |
    | Units         | days                                          |
    | Frequenc y    | YS                                            |
    | Standard Name | number_of_days_with_air_temperature_above_threshold |

    Args:
        tasmax: Daily maximum temperature (as an xarray DataArray).

    Returns:
        xarray.DataArray: Number of days per year with max temp > 40°C.
    """
    return xa.tx_days_above(tasmax, thresh="40.0 degC", freq="YS")  # type: ignore


def tn20(tasmin: xr.DataArray) -> xr.DataArray:
    """**Days with minimum temperature < -20°C**

    Average number of days per year with daily minimum temperature < -20°C.

    | Metadata      | Value                                         |
    |-------------- |-----------------------------------------------|
    | Identifier    | tn20                                          |
    | Units         | days                                          |
    | Frequency     | YS                                            |
    | Standard Name | number_of_days_with_air_temperature_below_threshold |

    Args:
        tasmin: Daily minimum temperature (as an xarray DataArray).

    Returns:
        xarray.DataArray: Number of days per year with min temp < -20°C.
    """
    return xa.tn_days_below(tasmin, thresh="-20.0 degC", freq="YS")  # type: ignore


"""Snow climatic indices"""


def solidprcptot_winter(pr: xr.DataArray, tas: xr.DataArray) -> xr.DataArray:
    """**Winter months accumulated solid precipitation (DJF)**

    Total accumulated solid precipitation during winter months (Dec-Jan-Feb).

    | Metadata      | Value                           |
    |-------------- |---------------------------------|
    | Identifier    | solidprcptot                    |
    | Units         | mm season-1                    |
    | Frequency     | QS-DEC                         |
    | Standard Name | lwe_thickness_of_snowfall_amount |

    Args:
        pr: Precipitation (as an xarray DataArray).
        tas: Air temperature (as an xarray DataArray).

    Returns:
        xarray.DataArray: Accumulated solid precipitation during winter months.
    """
    return xa.solid_precip_accumulation(pr=pr, tas=tas, thresh="0 degC", freq="QS-DEC")  # type: ignore


def solidprcptot_year(pr: xr.DataArray, tas: xr.DataArray) -> xr.DataArray:
    """**Annual accumulated solid precipitation**

    Total accumulated solid precipitation per year.

    | Metadata      | Value                           |
    |-------------- |---------------------------------|
    | Identifier    | solidprcptot                    |
    | Units         | mm year-1                      |
    | Frequency     | YS                             |
    | Standard Name | lwe_thickness_of_snowfall_amount |

    Args:
        pr: Precipitation (as an xarray DataArray).
        tas: Air temperature (as an xarray DataArray).

    Returns:
        xarray.DataArray: Annual accumulated solid precipitation.
    """
    return xa.solid_precip_accumulation(pr=pr, tas=tas, thresh="0 degC", freq="YS")  # type: ignore


"""Humidity climatic indices"""


def hurs_year(hurs: xr.DataArray) -> xr.DataArray:
    """**Annual mean relative humidity**

    Mean of daily relative humidity aggregated yearly.

    | Metadata      | Value           |
    |-------------- |-----------------|
    | Identifier    | hurs            |
    | Units         | %               |
    | Frequency     | YS              |
    | Standard Name | relative_humidity|

    Args:
        hurs: Relative humidity (as an xarray DataArray).

    Returns:
        xarray.DataArray: Annual mean relative humidity.
    """
    return hurs.resample(time="YS").mean()


def hi35(tas: xr.DataArray, hurs: xr.DataArray) -> xr.DataArray:
    """**Yearly days with heat index > 35°C**

    Number of days per year with heat index (perceived temperature) > 35°C.

    | Metadata      | Value                                         |
    |-------------- |-----------------------------------------------|
    | Identifier    | hi35                                          |
    | Units         | days                                          |
    | Frequency     | YS                                            |
    | Standard Name | number_of_days_with_air_temperature_above_threshold |

    Args:
        tas: Air temperature (as an xarray DataArray).
        hurs: Relative humidity (as an xarray DataArray).

    Returns:
        xarray.DataArray: Number of days with heat index > 35°C.
    """
    hi = xa.heat_index(tas=tas, hurs=hurs)
    return (hi > 35).resample(time="YS").sum(dim="time")  # type: ignore


def vpd(tas: xr.DataArray, hurs: xr.DataArray) -> xr.DataArray:
    """**Vapor pressure deficit**

    Annual vapor pressure deficit.

    | Metadata      | Value           |
    |-------------- |-----------------|
    | Identifier    | vpd             |
    | Units         | Pa              |
    | Frequency     | YS              |
    | Standard Name | vapor_pressure_deficit |

    Args:
        tas: Air temperature (as an xarray DataArray).
        hurs: Relative humidity (as an xarray DataArray).

    Returns:
        xarray.DataArray: Vapor pressure deficit.
    """
    return xa.vapor_pressure_deficit(tas=tas, hurs=hurs).resample(time="YS").mean()  # type: ignore
