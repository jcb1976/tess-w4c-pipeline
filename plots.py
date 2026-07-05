import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

#
# JELLYFISH PLOTS: FILTERED and SOLAR-CYCLE CORRECTED VERSIONS
#
def plot_tess_jellyfish(df, outfile, nbins_time=200, nbins_mag=200, sigma=1.0, site_name="TESS"):
    local = pd.to_datetime(df["local"])

    midnight = local.dt.floor("D")
    hours_from_midnight = (local - midnight).dt.total_seconds() / 3600.0

    hours_centered = hours_from_midnight.copy()
    hours_centered[hours_centered > 12] -= 24

    x = hours_centered.to_numpy(float)
    y = df["msas_mag_arcsec2"].astype(float).to_numpy()

    xmin = x.min() - 1.0
    xmax = x.max() + 1.0

    ymin, ymax = 17.0, 22.2

    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[nbins_time, nbins_mag],
        range=[[xmin, xmax], [ymin, ymax]]
    )

    H_smooth = gaussian_filter(H, sigma=sigma)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        H_smooth.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="inferno"
    )
    plt.colorbar(label="Counts")
    plt.xlabel("Hours from Local Midnight")
    plt.ylabel("Zenith Brightness [mag/arcsec^2]")
    plt.title(f"{site_name} Jellyfish Densitogram")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

#
# Histogram of filtered zenith brightness (single distribution)
#
def plot_filtered_histogram(df, outfile, bins=80, site_name="TESS"):
    mags = df["msas_mag_arcsec2"].astype(float)

    plt.figure(figsize=(8, 5))
    plt.hist(mags, bins=bins, color="steelblue", edgecolor="black")
    plt.xlabel("Zenith Brightness [mag/arcsec^2]")
    plt.ylabel("Counts")
    plt.title(f"Filtered Zenith Brightness ({site_name})")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

#
# Histogram comparing raw vs filtered values
#
def plot_raw_vs_filtered_histogram(raw_df, filtered_df, outfile, bins=80, site_name="TESS"):
    raw = raw_df["msas_mag_arcsec2"].astype(float)
    flt = filtered_df["msas_mag_arcsec2"].astype(float)

    plt.figure(figsize=(8, 5))
    plt.hist(raw, bins=bins, alpha=0.5, label="Raw", color="gray", edgecolor="black")
    plt.hist(flt, bins=bins, alpha=0.5, label="Filtered", color="steelblue", edgecolor="black")
    plt.xlabel("Zenith Brightness [mag/arcsec^2]")
    plt.ylabel("Counts")
    plt.title(f"Raw vs Filtered Zenith Brightness ({site_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

#
# Time series of filtered data
#
def plot_filtered_timeseries(df, outfile, site_name="TESS"):
    plt.figure(figsize=(12, 5))
    plt.plot(df["local"], df["msas_mag_arcsec2"], ".", markersize=2, color="steelblue")
    plt.xlabel("Local Time")
    plt.ylabel("Zenith Brightness [mag/arcsec^2]")
    plt.title(f"Filtered Zenith Brightness Time Series ({site_name})")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
