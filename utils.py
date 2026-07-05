import pandas as pd
import requests
from io import StringIO
from astropy.time import Time

def load_f107_from_web():
    url = "https://spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
    r = requests.get(url)
    r.raise_for_status()

    lines = r.text.splitlines()
    data_lines = []

    for ln in lines:
        parts = ln.strip().split()
        if len(parts) >= 7 and parts[0].isdigit() and len(parts[0]) == 8:
            data_lines.append(ln)

    df = pd.read_csv(StringIO("\n".join(data_lines)), delim_whitespace=True, header=None)
    df.columns = [
        "fluxdate", "fluxtime", "fluxjulian",
        "fluxcarrington", "fluxobsflux",
        "fluxadjflux", "fluxursi"
    ]

    df["fluxjulian"] = pd.to_numeric(df["fluxjulian"], errors="coerce")
    df = df.dropna(subset=["fluxjulian"])

    times = Time(df["fluxjulian"].values, format="jd")
    df["date"] = pd.Series(times.to_datetime()).dt.date

    return df[["date", "fluxobsflux"]].rename(columns={"fluxobsflux": "f107"})

