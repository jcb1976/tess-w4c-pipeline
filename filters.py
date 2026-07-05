import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body, GeocentricTrueEcliptic, AltAz
import astropy.units as u

def add_sun_moon_altitudes(df, site):
    # Resample to 5-minute intervals to balance compute time and resolution
    df_resampled = df.resample("5min").first()
    t = Time(df_resampled.index.values, scale="utc")

    altaz = AltAz(obstime=t, location=site)

    # Use get_body to avoid deprecation warnings
    sun = get_body("sun", t).transform_to(altaz)
    moon = get_body("moon", t).transform_to(altaz)

    df_resampled["sun_alt_deg"] = sun.alt.deg
    df_resampled["moon_alt_deg"] = moon.alt.deg

    # Merge the resampled calculations back into the high-resolution dataframe
    df = df.merge(df_resampled[["sun_alt_deg", "moon_alt_deg"]], left_index=True, right_index=True, how="left")
    
    # Fill in the gaps with the resampled calculations
    df[["sun_alt_deg", "moon_alt_deg"]] = df[["sun_alt_deg", "moon_alt_deg"]].ffill().bfill()

    return df

def add_cloud_flag(df, window_minutes=30, std_threshold=0.1):
    rolling_std = df["msas_mag_arcsec2"].rolling(f"{window_minutes}min").std()
    df["msas_std_rolling"] = rolling_std
    df["cloudy"] = rolling_std > std_threshold
    return df

def add_galactic_lat_zenith(df, site):
    df_5 = df.resample("5min").first()
    t = Time(df_5.index.values, scale="utc")
    jd = t.jd

    lat_deg = site.lat.deg
    lon_deg = site.lon.deg

    T = (jd - 2451545.0) / 36525.0
    GMST_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T**2
        - (T**3) / 38710000.0
    ) % 360

    LST_deg = (GMST_deg + lon_deg) % 360

    ra_zenith = LST_deg * u.deg
    dec_zenith = np.full_like(LST_deg, lat_deg) * u.deg

    zenith = SkyCoord(ra=ra_zenith, dec=dec_zenith, frame="icrs")
    gal = zenith.transform_to("galactic")

    df_5["b_zenith_deg"] = gal.b.deg
    df = df.merge(df_5[["b_zenith_deg"]], left_index=True, right_index=True, how="left")
    df["b_zenith_deg"] = df["b_zenith_deg"].ffill().bfill()
    return df

def add_zodiacal_flag(df, site):
    # Changed from 10min to 5min for consistency
    df_5 = df.resample("5min").first()
    t = Time(df_5.index.values, scale="utc")
    jd = t.jd

    lat_deg = site.lat.deg
    lon_deg = site.lon.deg

    T = (jd - 2451545.0) / 36525.0
    GMST_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T**2
        - (T**3) / 38710000.0
    ) % 360

    LST_deg = (GMST_deg + lon_deg) % 360

    ra_zenith = LST_deg * u.deg
    dec_zenith = np.full_like(LST_deg, lat_deg) * u.deg
    zenith = SkyCoord(ra=ra_zenith, dec=dec_zenith, frame="icrs")

    zenith_ecl = zenith.transform_to(GeocentricTrueEcliptic())
    df_5["beta_zenith_deg"] = zenith_ecl.lat.deg

    sun = get_body("sun", t)
    sep = sun.separation(zenith)
    df_5["solar_elong_deg"] = sep.deg

    beta_abs = df_5["beta_zenith_deg"].abs()
    eps = df_5["solar_elong_deg"]

    df_5["zodiacal"] = (beta_abs < 20) & (eps > 30) & (eps < 110)

    df = df.merge(df_5[["beta_zenith_deg", "solar_elong_deg", "zodiacal"]],
                  left_index=True, right_index=True, how="left")

    df[["beta_zenith_deg", "solar_elong_deg", "zodiacal"]] = (
        df[["beta_zenith_deg", "solar_elong_deg", "zodiacal"]].ffill().bfill()
    )
    df["zodiacal"] = df["zodiacal"].fillna(False).astype(bool)
    return df
