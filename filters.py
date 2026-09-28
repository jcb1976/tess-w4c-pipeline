import warnings
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body, GeocentricTrueEcliptic, AltAz
from astropy.coordinates.baseframe import NonRotationTransformationWarning

# Filter Astropy frame transformation warning
warnings.filterwarnings('ignore', category=NonRotationTransformationWarning)

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

def compute_rolling_chi2_5min(series_5min, window_points=19, poly_deg=2):
    """
    Computes the chi-squared (chi^2) statistic defined as the residual sum of squares (RSS)
    of a degree-2 polynomial fit over a 90-minute window (19 points at 5-minute cadence).
    """
    n = len(series_5min)
    vals = series_5min.values
    chi2_arr = np.zeros(n)
    
    half_w = window_points // 2
    
    # Pre-construct Vandermonde matrix and pseudoinverse for 19-point window acceleration
    x_full = np.linspace(-1, 1, window_points)
    V_full = np.vander(x_full, poly_deg + 1)
    V_pinv = np.linalg.pinv(V_full)
    
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        sub = vals[start:end]
        
        valid = ~np.isnan(sub)
        if np.sum(valid) < poly_deg + 2:
            chi2_arr[i] = 0.0
            continue
            
        if len(sub) == window_points and np.all(valid):
            coeffs = V_pinv @ sub
            fit = V_full @ coeffs
            res = sub - fit
            chi2_arr[i] = np.sum(res**2)
        else:
            x = np.linspace(-1, 1, len(sub))[valid]
            y = sub[valid]
            coeffs = np.polyfit(x, y, deg=poly_deg)
            fit = np.polyval(coeffs, x)
            chi2_arr[i] = np.sum((y - fit)**2)
            
    return pd.Series(chi2_arr, index=series_5min.index)

def add_cloud_flag(df, col="msas_mag_arcsec2", chi2_threshold=0.009, overcast_bright=None, overcast_dark=None, window_minutes=90):
    """
    Implements DSN chi-squared cloud detection with site-adaptive overcast ceilings:
    1. Chi-Squared Index (chi^2 >= chi2_threshold): Flags passing/variable clouds (works for both urban and dark sites).
    2. Urban Overcast Ceiling (MSAS <= overcast_bright): Flags bright overcast caused by reflected skyglow (e.g. <= 17.5 mag/arcsec^2).
    3. Dark-Sky Overcast Ceiling (MSAS >= overcast_dark): Flags dark overcast caused by blocked airglow (e.g. >= 22.1 mag/arcsec^2).
    """
    if col not in df.columns:
        if "msas1" in df.columns:
            col = "msas1"
            df["msas_mag_arcsec2"] = df["msas1"]
        else:
            df["chi2"] = 0.0
            df["cloudy"] = False
            return df

    # Resample to 5-minute cadence to standardize 19-point chi^2 windowing
    df_5 = df[[col]].resample("5min").mean()
    
    window_points = int(window_minutes // 5) + 1  # 90 / 5 + 1 = 19 points
    chi2_series_5 = compute_rolling_chi2_5min(df_5[col], window_points=window_points, poly_deg=2)
    
    df_5["chi2"] = chi2_series_5
    
    # Merge chi2 statistic back into original high-resolution dataframe
    df = df.merge(df_5[["chi2"]], left_index=True, right_index=True, how="left")
    df["chi2"] = df["chi2"].ffill().bfill().fillna(0.0)
    
    # Flag cloudiness
    # 1. Variable clouds: chi^2 >= threshold
    cloud_mask = df["chi2"] >= chi2_threshold
    
    # 2. Urban overcast ceiling: MSAS <= overcast_bright (reflected city light)
    if overcast_bright is not None:
        cloud_mask |= (df[col] <= overcast_bright)
        
    # 3. Dark-sky overcast ceiling: MSAS >= overcast_dark (blocked airglow)
    if overcast_dark is not None:
        cloud_mask |= (df[col] >= overcast_dark)
        
    df["cloudy"] = cloud_mask
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

    # Explicit transform_to('icrs') prevents NonRotationTransformationWarning
    sun = get_body("sun", t).transform_to("icrs")
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
