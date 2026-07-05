# TESS-W4C Data Reduction Pipeline

## Overview

This pipeline processes raw photometric data from the TESS-W4C four-channel night-sky brightness photometer. It ingests the raw `.dat` telemetry files, correctly handles UTC to local timezone conversions, calculates physically calibrated magnitudes and synthetic passbands, applies environmental and geometric filters, and generates a suite of analytical plots and a cleaned CSV dataset.

## The Data Processing Pipeline

The TESS-W4C records four independent channels: **UVIR750** (Channel 1), **UVIR650** (Channel 2), **RGB-R** (Channel 3), and **RGB-B** (Channel 4).

This script performs the following data reduction steps:

1. **Time Parsing:** Parses raw time strings as UTC and localizes them to the site-specific timezone defined in the configuration file.
2. **Native Magnitudes:** Recalculates the magnitudes for all four native channels directly from the raw frequencies and zero points (ZPs), applying a hard mask to reject saturated daytime readings (frequencies > 106,880 Hz).
3. **Synthetic Passbands (Flux Space):** * **RGB-G:** Computed by converting the broadband UVIR650, RGB-R, and RGB-B magnitudes into linear flux, isolating the green light via subtraction (`|f2 - f4 - f3|`), and converting back to magnitudes.
* **I-Band:** Computed in linear flux space by subtracting the UVIR750 flux from the UVIR650 flux (`|f2 - f1|`) to isolate the 650-750nm gap.


4. **Color Indices:** Calculated via straight arithmetic subtraction in logarithmic magnitude space: (B-G), (G-R), and (R-I).
5. **Data Purging:** Drops any record that results in unphysical math (NaNs) or falls outside the standard physical bounds for night-sky brightness (12.0 to 22.5 mag/arcsec²).

## Geometric & Environmental Filters

The pipeline uses `astropy` to calculate high-precision orbital mechanics for every row of data (resampled at 5-minute intervals for computational efficiency). These filters can be toggled on or off via command-line flags to isolate specific sky conditions:

* `--sun`: Removes data when the Sun is above -18° altitude (filters out daytime and all twilights).
* `--moon`: Removes data when the Moon is above -5° altitude.
* `--mw`: Removes data when the Galactic plane (Milky Way) is within 30° of the zenith.
* `--zod`: Removes data affected by Zodiacal light (ecliptic latitude < 20° and solar elongation between 30° and 110°).
* `--cloud`: Applies a 30-minute rolling standard deviation filter. High short-term variance in sky brightness indicates passing clouds reflecting urban light pollution; these periods are flagged and removed.

## Setup & Configuration

**Prerequisites:**
Ensure you have the following Python packages installed:
`pandas`, `numpy`, `matplotlib`, `astropy`, `pyyaml`

**Configuration File:**
The pipeline relies on a `tess_sites.yaml` file containing the physical coordinates and timezone for your deployment site. Example format:

```yaml
stars1429:
  latitude_deg: 32.2226
  longitude_deg: -110.9747
  elevation_m: 728.0
  timezone: "America/Phoenix"

```

**Data Directory:**
Raw `.dat` files must be placed in a subdirectory matching the site name (e.g., `data/stars1429/`).

## How to Run the Pipeline

You can run the pipeline directly via Python or by using the included bash wrapper script.

**Using the Bash Wrapper (Recommended):**
The `run_tess4c.sh` script provides a clean interface for executing the pipeline.

```bash
./run_tess4c.sh --instrument stars1429 --sun --moon --cloud

```

**Using Python Directly:**

```bash
python3 main4c.py --config tess_sites.yaml --site stars1429 --sun --moon --cloud

```

## Outputs

The pipeline outputs all files to the current working directory, prefixed with the site name.

* **Cleaned Data:** `<site>_4c_final.csv` (Contains only strictly validated, filtered nighttime records).
* **Individual Plots:** Histograms and 24-hour "jellyfish" scatter plots for every native channel, synthetic passband, and color index. Jellyfish plots are anti-aliased to exact 5-minute resolution bins.
* **Summary Panels:** * `<site>_summary_mag_hists.png`: A 6-panel grid of magnitude histograms.
* `<site>_summary_mag_jellyfish.png`: A 6-panel grid of magnitude jellyfish plots.
* `<site>_summary_indices.png`: A 6-panel grid of index histograms and jellyfish plots.
* `<site>_summary_color_color.png`: A 2-panel scatter plot of the (B-G) vs. (G-R) and (G-R) vs. (R-I) color spaces.
