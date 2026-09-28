import argparse
import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tess4c_io import load_tess4c_files
from filters import add_sun_moon_altitudes, add_galactic_lat_zenith, add_zodiacal_flag, add_cloud_flag

def compute_mag(freq, zp):
    """Calculates MSAS from Freq and ZP, masking saturation"""
    freq = pd.to_numeric(freq, errors='coerce')
    zp = pd.to_numeric(zp, errors='coerce')
    clean_freq = freq.mask((freq >= 106880.0) | (freq <= 0))
    return zp - 2.5 * np.log10(clean_freq)

def compute_synthetic_mag(flux_series):
    """Converts a flux series back into magnitudes, masking out exactly zero fluxes"""
    clean_flux = flux_series.mask(flux_series <= 0)
    return -2.5 * np.log10(clean_flux)

# --- Individual Custom Plotters (Magnitudes) ---
def plot_custom_mag_histogram(df, ch, label, outfile, site_name, filter_str):
    df_plot = df.dropna(subset=[ch])
    if df_plot.empty: return
    t_min = df_plot.index.min().strftime('%Y-%m-%d')
    t_max = df_plot.index.max().strftime('%Y-%m-%d')
    n_pts = len(df_plot)
    date_n_str = f"Dates: {t_min} to {t_max} | N = {n_pts:,}"

    plt.figure(figsize=(8, 5))
    plt.hist(df_plot[ch], bins=60, range=(12, 22.5), color='teal', edgecolor='black', alpha=0.8)
    plt.title(f"Histogram: {site_name} {label}\n{date_n_str}\n{filter_str}", fontsize=10)
    plt.xlabel("Mag/arcsec^2")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_custom_mag_jellyfish(df, ch, label, outfile, site_name, filter_str):
    df_plot = df.dropna(subset=[ch]).copy()
    if df_plot.empty: return
    t_min = df_plot.index.min().strftime('%Y-%m-%d')
    t_max = df_plot.index.max().strftime('%Y-%m-%d')
    n_pts = len(df_plot)
    date_n_str = f"Dates: {t_min} to {t_max} | N = {n_pts:,}"

    hours = df_plot.index.hour + df_plot.index.minute / 60.0
    hours = np.where(hours > 12, hours - 24, hours)
    
    my_cmap = plt.get_cmap("inferno").copy()
    my_cmap.set_bad("black")
    
    plt.figure(figsize=(8, 5))
    plt.hist2d(hours, df_plot[ch], bins=[288, 100], range=[[-12, 12], [16.0, 22.5]], cmap=my_cmap, norm=LogNorm())
    plt.colorbar(label="Count (Log Scale)")
    plt.title(f"Jellyfish: {site_name} {label}\n{date_n_str}\n{filter_str}", fontsize=10)
    plt.xlabel("Hours from Local Midnight")
    plt.ylabel("Mag/arcsec^2")
    plt.ylim(16.0, 22.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

# --- Individual Custom Plotters (Indices) ---
def plot_custom_index_histogram(df, ch, label, outfile, site_name, filter_str):
    df_plot = df.dropna(subset=[ch])
    if df_plot.empty: return
    t_min = df_plot.index.min().strftime('%Y-%m-%d')
    t_max = df_plot.index.max().strftime('%Y-%m-%d')
    n_pts = len(df_plot)
    date_n_str = f"Dates: {t_min} to {t_max} | N = {n_pts:,}"

    plt.figure(figsize=(8, 5))
    plt.hist(df_plot[ch], bins=60, range=(-3, 3), color='teal', edgecolor='black', alpha=0.8)
    plt.title(f"Histogram: {site_name} {label}\n{date_n_str}\n{filter_str}", fontsize=10)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_custom_index_jellyfish(df, ch, label, outfile, site_name, filter_str):
    df_plot = df.dropna(subset=[ch]).copy()
    if df_plot.empty: return
    t_min = df_plot.index.min().strftime('%Y-%m-%d')
    t_max = df_plot.index.max().strftime('%Y-%m-%d')
    n_pts = len(df_plot)
    date_n_str = f"Dates: {t_min} to {t_max} | N = {n_pts:,}"

    hours = df_plot.index.hour + df_plot.index.minute / 60.0
    hours = np.where(hours > 12, hours - 24, hours)
    
    my_cmap = plt.get_cmap("inferno").copy()
    my_cmap.set_bad("black")
    
    plt.figure(figsize=(8, 5))
    plt.hist2d(hours, df_plot[ch], bins=[288, 100], range=[[-12, 12], [-3, 3]], cmap=my_cmap, norm=LogNorm())
    plt.colorbar(label="Count (Log Scale)")
    plt.title(f"Jellyfish: {site_name} {label}\n{date_n_str}\n{filter_str}", fontsize=10)
    plt.xlabel("Hours from Local Midnight")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

# --- Multi-Panel Summary Plotters ---
def plot_multipanel_mag_hists(df, site_name, outdir, filter_str):
    if df.empty: return
    t_min = df.index.min().strftime('%Y-%m-%d')
    t_max = df.index.max().strftime('%Y-%m-%d')
    n_pts = len(df)
    date_n_str = f"Dates: {t_min} to {t_max} | Total N = {n_pts:,}"

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{site_name} Magnitude Histograms\n{date_n_str}\n{filter_str}", fontsize=13, y=0.98)
    axes = axes.flatten()
    cols = ['msas2', 'msas1', 'msas4', 'RGB-G', 'msas3', 'I']
    names = ['UVIR650', 'UVIR750', 'RGB-B', 'RGB-G', 'RGB-R', 'I-band']
    
    for ax, ch, name in zip(axes, cols, names):
        if ch in df.columns and not df[ch].dropna().empty:
            sub = df[ch].dropna()
            ax.hist(sub, bins=60, range=(12, 22.5), color='teal', edgecolor='black', alpha=0.8)
            ax.set_title(f"{name} (N={len(sub):,})", fontsize=11)
            ax.set_xlabel("Mag/arcsec^2")
            ax.set_ylabel("Count")
            ax.grid(True, linestyle='--', alpha=0.5)
            
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(outdir, f"{site_name}_summary_mag_hists.png"), dpi=150)
    plt.close()

def plot_multipanel_mag_jellyfish(df, site_name, outdir, filter_str):
    if df.empty: return
    t_min = df.index.min().strftime('%Y-%m-%d')
    t_max = df.index.max().strftime('%Y-%m-%d')
    n_pts = len(df)
    date_n_str = f"Dates: {t_min} to {t_max} | Total N = {n_pts:,}"

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{site_name} Magnitude Jellyfish\n{date_n_str}\n{filter_str}", fontsize=13, y=0.98)
    axes = axes.flatten()
    cols = ['msas2', 'msas1', 'msas4', 'RGB-G', 'msas3', 'I']
    names = ['UVIR650', 'UVIR750', 'RGB-B', 'RGB-G', 'RGB-R', 'I-band']
    
    hours = df.index.hour + df.index.minute / 60.0
    hours = np.where(hours > 12, hours - 24, hours)
    
    my_cmap = plt.get_cmap("inferno").copy()
    my_cmap.set_bad("black")
    
    for ax, ch, name in zip(axes, cols, names):
        if ch in df.columns and not df[ch].dropna().empty:
            sub = df.dropna(subset=[ch])
            sub_hours = hours[df[ch].notna()]
            h, _, _, im = ax.hist2d(sub_hours, sub[ch], bins=[288, 100], range=[[-12, 12], [16.0, 22.5]], cmap=my_cmap, norm=LogNorm())
            fig.colorbar(im, ax=ax, label="Count (Log)")
            ax.set_title(f"{name} (N={len(sub):,})", fontsize=11)
            ax.set_xlabel("Hours from Local Midnight")
            ax.set_ylabel("Mag/arcsec^2")
            ax.set_ylim(16.0, 22.5)
            
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(outdir, f"{site_name}_summary_mag_jellyfish.png"), dpi=150)
    plt.close()

def plot_multipanel_indices(df, site_name, outdir, filter_str):
    if df.empty: return
    t_min = df.index.min().strftime('%Y-%m-%d')
    t_max = df.index.max().strftime('%Y-%m-%d')
    n_pts = len(df)
    date_n_str = f"Dates: {t_min} to {t_max} | Total N = {n_pts:,}"

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle(f"{site_name} Color Indices\n{date_n_str}\n{filter_str}", fontsize=13, y=0.98)
    cols = ['idx_BG', 'idx_GR', 'idx_RI']
    names = ['B-G', 'G-R', 'R-I']
    
    hours = df.index.hour + df.index.minute / 60.0
    hours = np.where(hours > 12, hours - 24, hours)
    
    my_cmap = plt.get_cmap("inferno").copy()
    my_cmap.set_bad("black")
    
    for i, (ch, name) in enumerate(zip(cols, names)):
        if ch in df.columns and not df[ch].dropna().empty:
            sub = df.dropna(subset=[ch])
            sub_hours = hours[df[ch].notna()]
            
            # Left Column: Histograms
            axes[i, 0].hist(sub[ch], bins=60, range=(-3, 3), color='teal', edgecolor='black', alpha=0.8)
            axes[i, 0].set_title(f"{name} Histogram (N={len(sub):,})", fontsize=11)
            axes[i, 0].set_xlabel("Index Value")
            axes[i, 0].set_ylabel("Count")
            axes[i, 0].grid(True, linestyle='--', alpha=0.5)
            
            # Right Column: Jellyfish
            h, _, _, im = axes[i, 1].hist2d(sub_hours, sub[ch], bins=[288, 100], range=[[-12, 12], [-3, 3]], cmap=my_cmap, norm=LogNorm())
            fig.colorbar(im, ax=axes[i, 1], label="Count (Log)")
            axes[i, 1].set_title(f"{name} Jellyfish (N={len(sub):,})", fontsize=11)
            axes[i, 1].set_xlabel("Hours from Midnight")
            axes[i, 1].set_ylabel("Index Value")
            
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(outdir, f"{site_name}_summary_indices.png"), dpi=150)
    plt.close()

def plot_multipanel_color_color(df, site_name, outdir, filter_str):
    if df.empty: return
    t_min = df.index.min().strftime('%Y-%m-%d')
    t_max = df.index.max().strftime('%Y-%m-%d')
    n_pts = len(df)
    date_n_str = f"Dates: {t_min} to {t_max} | Total N = {n_pts:,}"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{site_name} Color-Color Diagrams\n{date_n_str}\n{filter_str}", fontsize=13, y=0.98)
    pairs = [('idx_BG', 'idx_GR', 'B-G', 'G-R'), ('idx_GR', 'idx_RI', 'G-R', 'R-I')]
    
    for ax, (x, y, xname, yname) in zip(axes, pairs):
        if x in df.columns and y in df.columns:
            sub = df.dropna(subset=[x, y])
            ax.scatter(sub[x], sub[y], s=1, alpha=0.1, color='teal')
            ax.set_title(f"{xname} vs {yname} (N={len(sub):,})", fontsize=11)
            ax.set_xlabel(xname)
            ax.set_ylabel(yname)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.grid(True, linestyle='--', alpha=0.5)
            
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(os.path.join(outdir, f"{site_name}_summary_color_color.png"), dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--sun", action="store_true")
    parser.add_argument("--sun_alt", type=float, default=-18.0, help="Sun altitude threshold in degrees (default: -18.0)")
    parser.add_argument("--moon", action="store_true")
    parser.add_argument("--moon_alt", type=float, default=-10.0, help="Moon altitude threshold in degrees (default: -10.0)")
    parser.add_argument("--mw", action="store_true")
    parser.add_argument("--zod", action="store_true")
    parser.add_argument("--cloud", action="store_true")
    parser.add_argument("--chi2_threshold", type=float, default=0.009, help="Chi-squared threshold for variable cloud detection (default: 0.009)")
    parser.add_argument("--overcast_bright", type=float, default=None, help="Urban overcast ceiling: flags MSAS <= threshold (e.g., 17.5 for stars1429)")
    parser.add_argument("--overcast_dark", type=float, default=None, help="Dark-sky overcast ceiling: flags MSAS >= threshold (e.g., 22.1 for Spring Valley)")
    args = parser.parse_args()

    # Build dynamic filter string for plots
    filter_texts = []
    if args.sun:   filter_texts.append(f"Sun (<{args.sun_alt} deg)")
    if args.moon:  filter_texts.append(f"Moon (<{args.moon_alt} deg)")
    if args.mw:    filter_texts.append("Milky Way (>30 deg)")
    if args.zod:   filter_texts.append("Zodiacal (Off)")
    if args.cloud: 
        cloud_desc = f"Clouds (chi2>={args.chi2_threshold}"
        if args.overcast_bright is not None:
            cloud_desc += f", bright<={args.overcast_bright}"
        if args.overcast_dark is not None:
            cloud_desc += f", dark>={args.overcast_dark}"
        cloud_desc += ")"
        filter_texts.append(cloud_desc)
    
    filter_str = "Filters: " + ", ".join(filter_texts) if filter_texts else "Filters: None"

    outdir = "output"
    os.makedirs(outdir, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    from astropy.coordinates import EarthLocation
    s = cfg[args.site]
    site = EarthLocation(lat=s["latitude_deg"], lon=s["longitude_deg"], height=s.get("elevation_m", 0.0))
    
    # 1. Ingestion
    local_tz = s.get("timezone", "America/Phoenix")
    df_raw = load_tess4c_files(os.path.join("data", args.site), local_tz=local_tz)
    n_raw_total = len(df_raw)
    
    # 2. Compute Native Magnitudes
    print("Recalculating native magnitudes from frequencies...")
    df_raw['msas1'] = compute_mag(df_raw['f1'], 19.9)
    df_raw['msas2'] = compute_mag(df_raw['f2'], df_raw['zp2'])
    df_raw['msas3'] = compute_mag(df_raw['f3'], df_raw['zp3'])
    df_raw['msas4'] = compute_mag(df_raw['f4'], df_raw['zp4'])

    # Step 0: Sensor Integrity / Range Filter (mask out non-physical ranges)
    valid_sensor_mask = pd.Series(True, index=df_raw.index)
    for ch in ['msas1', 'msas2', 'msas3', 'msas4']:
        valid_sensor_mask &= (df_raw[ch] > 10.0) & (df_raw[ch] < 22.5)
        
    df_valid = df_raw[valid_sensor_mask].copy()
    n_valid_sensor = len(df_valid)
    n_screened_sensor = n_raw_total - n_valid_sensor
        
    # 3. Geometry & Flags
    print("Computing Sun/Moon geometry and flags on valid dataset...")
    df_valid = add_sun_moon_altitudes(df_valid, site)
    
    if args.mw:    df_valid = add_galactic_lat_zenith(df_valid, site)
    if args.zod:   df_valid = add_zodiacal_flag(df_valid, site)
    if args.cloud: 
        df_valid["msas_mag_arcsec2"] = df_valid["msas1"]
        df_valid = add_cloud_flag(
            df_valid, 
            chi2_threshold=args.chi2_threshold, 
            overcast_bright=args.overcast_bright, 
            overcast_dark=args.overcast_dark
        )

    # 4. Sequential Filtering Cascade Tracking
    # Step a: Sun Filter
    if args.sun and "sun_alt_deg" in df_valid.columns:
        step1_df = df_valid[df_valid["sun_alt_deg"] < args.sun_alt].copy()
    else:
        step1_df = df_valid.copy()
    n_step1 = len(step1_df)
    n_screened_step1 = n_valid_sensor - n_step1

    # Step b: Moon Filter
    if args.moon and "moon_alt_deg" in step1_df.columns:
        step2_df = step1_df[step1_df["moon_alt_deg"] < args.moon_alt].copy()
    else:
        step2_df = step1_df.copy()
    n_step2 = len(step2_df)
    n_screened_step2 = n_step1 - n_step2

    # Step c: Cloud Filter
    if args.cloud and "cloudy" in step2_df.columns:
        step3_df = step2_df[~step2_df["cloudy"]].copy()
    else:
        step3_df = step2_df.copy()
    n_step3 = len(step3_df)
    n_screened_step3 = n_step2 - n_step3

    # Step d: Milky Way Filter
    if args.mw and "b_zenith_deg" in step3_df.columns:
        step4_df = step3_df[step3_df["b_zenith_deg"].abs() > 30.0].copy()
    else:
        step4_df = step3_df.copy()
    n_step4 = len(step4_df)
    n_screened_step4 = n_step3 - n_step4

    # Step e: Zodiacal Light Filter
    if args.zod and "zodiacal" in step4_df.columns:
        step5_df = step4_df[~step4_df["zodiacal"]].copy()
    else:
        step5_df = step4_df.copy()
    n_step5 = len(step5_df)
    n_screened_step5 = n_step4 - n_step5

    df_filtered = step5_df.copy()

    # 5. Synthetic Passbands & Color Indices
    print("Computing synthetic passbands and color indices...")
    f1_flux = 10 ** (df_filtered['msas1'] / -2.5)
    f2_flux = 10 ** (df_filtered['msas2'] / -2.5)
    f3_flux = 10 ** (df_filtered['msas3'] / -2.5)
    f4_flux = 10 ** (df_filtered['msas4'] / -2.5)
    
    df_filtered['RGB-G'] = compute_synthetic_mag((f2_flux - f4_flux - f3_flux).abs())
    df_filtered['I'] = compute_synthetic_mag((f2_flux - f1_flux).abs())
    
    df_filtered['idx_BG'] = df_filtered['msas4'] - df_filtered['RGB-G']
    df_filtered['idx_GR'] = df_filtered['RGB-G'] - df_filtered['msas3']
    df_filtered['idx_RI'] = df_filtered['msas3'] - df_filtered['I']

    export_cols = ['msas1', 'msas2', 'msas3', 'msas4', 'RGB-G', 'I', 'idx_BG', 'idx_GR', 'idx_RI']
    df_filtered = df_filtered.dropna(subset=export_cols).copy()

    # Output Filtering Cascade Summary Table
    pct_valid = 100.0 if n_valid_sensor > 0 else 0.0
    pct1 = (n_step1 / n_valid_sensor * 100.0) if n_valid_sensor > 0 else 0.0
    pct2 = (n_step2 / n_valid_sensor * 100.0) if n_valid_sensor > 0 else 0.0
    pct3 = (n_step3 / n_valid_sensor * 100.0) if n_valid_sensor > 0 else 0.0
    pct4 = (n_step4 / n_valid_sensor * 100.0) if n_valid_sensor > 0 else 0.0
    pct5 = (n_step5 / n_valid_sensor * 100.0) if n_valid_sensor > 0 else 0.0

    cloud_desc_str = f"chi^2 < {args.chi2_threshold}"
    if args.overcast_bright is not None: cloud_desc_str += f" & bright > {args.overcast_bright}"
    if args.overcast_dark is not None:   cloud_desc_str += f" & dark < {args.overcast_dark}"

    print("\n" + "="*68)
    print("DATA FILTERING CASCADE & CHI-SQUARED CLOUD SUMMARY")
    print("="*68)
    print(f"1. Total Raw Measurements Ingested:             {n_raw_total:>8d}")
    print(f"   - Invalid / Saturated Sensor Readings Out:    {n_screened_sensor:>8d}")
    print(f"   = Valid Sensor Measurements Remaining:         {n_valid_sensor:>8d}  ({pct_valid:5.1f}%)")
    print("-" * 68)
    print("2. Sequential Filtering Cascade:")
    print(f"   a. Astronomical Night Filter (Sun <= {args.sun_alt} deg):")
    print(f"      - Screened Out (Daytime & Twilight):       {n_screened_step1:>8d}")
    print(f"      = Remaining (Astronomical Night):          {n_step1:>8d}  ({pct1:5.1f}%)")
    print(f"   b. Lunar Altitude Filter (Moon < {args.moon_alt} deg):")
    print(f"      - Screened Out (Moonlit Periods):          {n_screened_step2:>8d}")
    print(f"      = Remaining (Moonless Astronomical Night): {n_step2:>8d}  ({pct2:5.1f}%)")
    print(f"   c. Chi-Squared Cloud Filter ({cloud_desc_str}):")
    print(f"      - Screened Out (Variable & Overcast Clouds):{n_screened_step3:>8d}")
    print(f"      = Remaining (Cloud-Free Night):            {n_step3:>8d}  ({pct3:5.1f}%)")
    print(f"   d. Galactic Plane Filter (|b_zenith| >= 30.0 deg):")
    print(f"      - Screened Out (Milky Way in Zenith):      {n_screened_step4:>8d}")
    print(f"      = Remaining (Milky Way Clear):             {n_step4:>8d}  ({pct4:5.1f}%)")
    print(f"   e. Zodiacal Light Filter (|beta_zenith| >= 20.0 deg):")
    print(f"      - Screened Out (Zodiacal Light in Zenith): {n_screened_step5:>8d}")
    print(f"      = Remaining (Zodiacal Clear):              {n_step5:>8d}  ({pct5:5.1f}%)")
    print("-" * 68)
    print(f"3. Final Clean Dark-Sky Measurements Retained:  {n_step5:>8d}  ({pct5:5.1f}%)")
    print("="*68 + "\n")

    # 6. Plotting
    print("Generating individual plots...")
    mag_labels = {
        'msas1': 'UVIR750', 'msas2': 'UVIR650', 'msas3': 'RGB-R', 'msas4': 'RGB-B',
        'RGB-G': 'RGB-G', 'I': 'I-band'
    }
    for ch, label in mag_labels.items():
        if ch in df_filtered.columns:
            plot_df = df_filtered[[ch]].copy()
            plot_df['local'] = df_filtered.index.tz_localize(None)
            
            if not plot_df.empty:
                outfile = os.path.join(outdir, f"{args.site}_{label}_hist.png")
                plot_custom_mag_histogram(plot_df, ch, label, outfile, args.site, filter_str)
                
                outfile = os.path.join(outdir, f"{args.site}_{label}_jellyfish.png")
                plot_custom_mag_jellyfish(plot_df, ch, label, outfile, args.site, filter_str)
                
    idx_labels = {'idx_BG': 'B-G', 'idx_GR': 'G-R', 'idx_RI': 'R-I'}
    for ch, label in idx_labels.items():
        if ch in df_filtered.columns:
            plot_df = df_filtered[[ch]].copy()
            plot_df['local'] = df_filtered.index.tz_localize(None)
            
            if not plot_df.empty:
                outfile = os.path.join(outdir, f"{args.site}_{label}_hist.png")
                plot_custom_index_histogram(plot_df, ch, label, outfile, args.site, filter_str)
                
                outfile = os.path.join(outdir, f"{args.site}_{label}_jellyfish.png")
                plot_custom_index_jellyfish(plot_df, ch, label, outfile, args.site, filter_str)
    
    # 7. Summary Plotting
    print("Generating multi-panel summary plots...")
    plot_multipanel_mag_hists(df_filtered, args.site, outdir, filter_str)
    plot_multipanel_mag_jellyfish(df_filtered, args.site, outdir, filter_str)
    plot_multipanel_indices(df_filtered, args.site, outdir, filter_str)
    plot_multipanel_color_color(df_filtered, args.site, outdir, filter_str)
    
    df_filtered[export_cols].to_csv(os.path.join(outdir, f"{args.site}_4c_final.csv"))
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
