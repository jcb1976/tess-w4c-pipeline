import pandas as pd
import numpy as np


def compute_nightly(df):
    grouped = df.groupby("date_local")

    nightly = grouped.agg(
        msas_median=("msas_mag_arcsec2", "median"),
        sky_temp_median=("sky_temp_C", "median"),
        enclosure_temp_median=("enclosure_temp_C", "median"),
        n_points=("msas_mag_arcsec2", "count"),
    ).reset_index()

    return nightly


def analyze_nightly_sky_brightness(nightly):
    nightly = nightly.copy()

    x = pd.to_datetime(nightly["date_local"])
    x_num = (x - x.min()).dt.days.astype(float)

    y = nightly["msas_median"].astype(float)

    coeffs = np.polyfit(x_num, y, 1)
    trend = np.polyval(coeffs, x_num)

    nightly["msas_corrected"] = y - (trend - trend.mean())

    return nightly, coeffs, trend


def expand_corrected_to_timestamps(df, nightly):
    df = df.copy()

    nightly_map = nightly.set_index("date_local")["msas_corrected"]

    df["msas_corrected"] = df["date_local"].map(nightly_map)

    return df

