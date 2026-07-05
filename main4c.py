import argparse
import yaml
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
def plot_custom_mag_histogram(df, ch, label, outfile, site_name):
    df_plot = df.dropna(subset=[ch])
    if df_plot.empty: return
    plt.figure(figsize=(8, 5))
    plt.hist(df_plot[ch], bins=60, range=(12, 22.5), color='teal', edgecolor='black', alpha=0.8)
    plt.title(f"Histogram: {site_name} {label}")
    plt.xlabel("Mag/arcsec^2")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_custom_mag_jellyfish(df, ch, label, outfile, site_name):
    df_plot = df.dropna(subset=[ch]).copy()
    if df_plot.empty: return
    hours = df_plot.index.hour + df_plot.index.minute / 60.0
    hours = np.where(hours > 12, hours - 24, hours)
    plt.figure(figsize=(8, 5))
    # 288 bins = exactly 5 minutes per pixel over 24 hours
    plt.hist2d(hours, df_plot[ch], bins=[288, 100], range=[[-12, 12], [12, 22.5]], cmap="inferno")
    plt.colorbar(label="Count")
    plt.title(f"Jellyfish: {site_name} {label}")
    plt.xlabel("Hours from Local Midnight")
    plt.ylabel("Mag/arcsec^2")
    plt.savefig(outfile, dpi=150)
    plt.close()

# --- Individual Custom Plotters (Indices) ---
def plot_custom_index_histogram(df, ch, label, outfile, site_name):
    df_plot = df.dropna(subset=[ch])
    if df_plot.empty: return
    plt.figure(figsize=(8, 5))
    plt.hist(df_plot[ch], bins=60, range=(-3, 3), color='teal', edgecolor='black', alpha=0.8)
    plt.title(f"Histogram: {site_name} {label}")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_custom_index_jellyfish(df, ch, label, outfile, site_name):
    df_plot = df.dropna(subset=[ch]).copy()
    if df_plot.empty: return
    hours = df_plot.index.hour + df_plot.index.minute / 60.0
    hours = np.where(hours > 12, hours - 24, hours)
    plt.figure(figsize=(8, 5))
    plt.hist2d(hours, df_plot[ch], bins=[288, 100], range=[[-12, 12], [-3, 3]], cmap="inferno")
    plt.colorbar(label="Value")
    plt.title(f"Jellyfish: {site_name} {label}")
    plt.xlabel("Hours from Local Midnight")
    plt.ylabel("Value")
    plt.savefig(outfile, dpi=150)
    plt.close()

# --- New Multi-Panel Summary Plotters ---
def plot_multipanel_mag_hists(df, site_name):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    cols = ['msas2', 'msas1', 'msas4', 'RGB-G', 'msas3', 'I']
    names = ['UVIR650', 'UVIR750', 'RGB-B', 'RGB-G', 'RGB-R', 'I-band']
    
    for ax, ch, name in zip(axes, cols, names):
        if ch in df.columns and not df[ch].dropna().empty:
            ax.hist(df[ch].dropna(), bins=60, range=(12, 22.5), color='teal', edgecolor='black', alpha=0.8)
            ax.set_title(f"{site_name} {name}")
            ax.set_xlabel("Mag/arcsec^2")
            ax.set_ylabel("Count")
            ax.grid(True, linestyle='--', alpha=0.5)
            
    plt.tight_layout()
    plt.savefig(f"{site_name}_summary_mag_hists.png", dpi=150)
    plt.close()

def plot_multipanel_mag_jellyfish(df, site_name):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    cols = ['msas2', 'msas1', 'msas4', 'RGB-G', 'msas3', 'I']
    names = ['UVIR650', 'UVIR750', 'RGB-B', 'RGB-G', 'RGB-R', 'I-band']
    
    hours = df.index.hour + df.index.minute / 60.0
    hours = np.where(hours > 12, hours - 24, hours)
    
    for ax, ch, name in zip(axes, cols, names):
        if ch in df.columns and not df[ch].dropna().empty:
            h, _, _, im = ax.hist2d(hours, df[ch], bins=[288, 100], range=[[-12, 12], [12, 22.5]], cmap="inferno")
            fig.colorbar(im, ax=ax, label="Count")
            ax.set_title(f"{site_name} {name}")
            ax.set_xlabel("Hours from Local Midnight")
            ax.set_ylabel("Mag/arcsec^2")
            
    plt.tight_layout()
    plt.savefig(f"{site_name}_summary_mag_jellyfish.png", dpi=150)
    plt.close()

def plot_multipanel_indices(df, site_name):
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    cols = ['idx_BG', 'idx_GR', 'idx_RI']
    names = ['B-G', 'G-R', 'R-I']
    
    hours = df.index.hour + df.index.minute / 60.0
    hours = np.where(hours > 12, hours - 24, hours)
    
    for i, (ch, name) in enumerate(zip(cols, names)):
        if ch in df.columns and not df[ch].dropna().empty:
            # Left Column: Histograms
            axes[i, 0].hist(df[ch], bins=60, range=(-3, 3), color='teal', edgecolor='black', alpha=0.8)
            axes[i, 0].set_title(f"{site_name} {name} Histogram")
            axes[i, 0].set_xlabel("Index Value")
            axes[i, 0].set_ylabel("Count")
            axes[i, 0].grid(True, linestyle='--', alpha=0.5)
            
            # Right Column: Jellyfish
            h, _, _, im = axes[i, 1].hist2d(hours, df[ch], bins=[288, 100], range=[[-12, 12], [-3, 3]], cmap="inferno")
            fig.colorbar(im, ax=axes[i, 1], label="Count")
            axes[i, 1].set_title(f"{site_name} {name} Jellyfish")
            axes[i, 1].set_xlabel("Hours from Midnight")
            axes[i, 1].set_ylabel("Index Value")
            
    plt.tight_layout()
    plt.savefig(f"{site_name}_summary_indices.png", dpi=150)
    plt.close()

def plot_multipanel_color_color(df, site_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pairs = [('idx_BG', 'idx_GR', 'B-G', 'G-R'), ('idx_GR', 'idx_RI', 'G-R', 'R-I')]
    
    for ax, (x, y, xname, yname) in zip(axes, pairs):
        if x in df.columns and y in df.columns:
            ax.scatter(df[x], df[y], s=1, alpha=0.1, color='teal')
            ax.set_title(f"{site_name} Color-Color: {xname} vs {yname}")
            ax.set_xlabel(xname)
            ax.set_ylabel(yname)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.grid(True, linestyle='--', alpha=0.5)
            
    plt.tight_layout()
    plt.savefig(f"{site_name}_summary_color_color.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--sun", action="store_true")
    parser.add_argument("--moon", action="store_true")
    parser.add_argument("--mw", action="store_true")
    parser.add_argument("--zod", action="store_true")
    parser.add_argument("--cloud", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    from astropy.coordinates import EarthLocation
    s = cfg[args.site]
    site = EarthLocation(lat=s["latitude_deg"], lon=s["longitude_deg"], height=s.get("elevation_m", 0.0))
    
    # 1. Strict Ingestion
    local_tz = s.get("timezone", "America/Phoenix")
    df = load_tess4c_files(os.path.join("data", args.site), local_tz=local_tz)
    
    # 2. Compute Native Magnitudes
    print("Recalculating native magnitudes from frequencies...")
    df['msas1'] = compute_mag(df['f1'], 19.9)
    df['msas2'] = compute_mag(df['f2'], df['zp2'])
    df['msas3'] = compute_mag(df['f3'], df['zp3'])
    df['msas4'] = compute_mag(df['f4'], df['zp4'])
        
    # 3. Geometry 
    print("Computing Sun/Moon geometry and flags on full dataset...")
    df = add_sun_moon_altitudes(df, site)
    
    if args.mw:    df = add_galactic_lat_zenith(df, site)
    if args.zod:   df = add_zodiacal_flag(df, site)
    if args.cloud: 
        df["msas_mag_arcsec2"] = df["msas1"]
        df = add_cloud_flag(df)

    # 4. Apply Physical & Geometric Filters
    mask = pd.Series(True, index=df.index)
    for ch in ['msas1', 'msas2', 'msas3', 'msas4']:
        mask &= (df[ch] > 10.0) & (df[ch] < 22.5)
    
    if args.sun:   mask &= (df["sun_alt_deg"] < -18)
    if args.moon:  mask &= (df["moon_alt_deg"] < -5)
    if args.mw:    mask &= (df["b_zenith_deg"].abs() > 30)
    if args.zod:   mask &= (~df["zodiacal"])
    if args.cloud: mask &= (~df["cloudy"])
    
    df_filtered = df[mask].copy()

    # 5. SYNTHETIC PASSBANDS & COLOR INDICES
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

    # DROP NANS: Purge any row that failed a math check
    export_cols = ['msas1', 'msas2', 'msas3', 'msas4', 'RGB-G', 'I', 'idx_BG', 'idx_GR', 'idx_RI']
    df_filtered = df_filtered.dropna(subset=export_cols).copy()
    
    print(f"Records remaining for plotting and export: {len(df_filtered)}")

    # 6. PLOTTING (Original Individual Plots)
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
                plot_custom_mag_histogram(plot_df, ch, label, f"{args.site}_{label}_hist.png", site_name=args.site)
                plot_custom_mag_jellyfish(plot_df, ch, label, f"{args.site}_{label}_jellyfish.png", site_name=args.site)
                
    idx_labels = {'idx_BG': 'B-G', 'idx_GR': 'G-R', 'idx_RI': 'R-I'}
    for ch, label in idx_labels.items():
        if ch in df_filtered.columns:
            plot_df = df_filtered[[ch]].copy()
            plot_df['local'] = df_filtered.index.tz_localize(None)
            if not plot_df.empty:
                plot_custom_index_histogram(plot_df, ch, label, f"{args.site}_{label}_hist.png", site_name=args.site)
                plot_custom_index_jellyfish(plot_df, ch, label, f"{args.site}_{label}_jellyfish.png", site_name=args.site)
    
    for x, y in [('idx_BG', 'idx_GR'), ('idx_GR', 'idx_RI')]:
        plt.figure(figsize=(6, 6))
        plt.scatter(df_filtered[x], df_filtered[y], s=1, alpha=0.1, color='teal')
        plt.xlabel(idx_labels.get(x, x))
        plt.ylabel(idx_labels.get(y, y))
        plt.title(f"{args.site} Color-Color: {idx_labels.get(x, x)} vs {idx_labels.get(y, y)}")
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(f"{args.site}_cc_{x}_{y}.png", dpi=150)
        plt.close()
        
    # 7. NEW SUMMARY PLOTTING (Multi-panel)
    print("Generating multi-panel summary plots...")
    plot_multipanel_mag_hists(df_filtered, args.site)
    plot_multipanel_mag_jellyfish(df_filtered, args.site)
    plot_multipanel_indices(df_filtered, args.site)
    plot_multipanel_color_color(df_filtered, args.site)
    
    # Save Final CSV
    df_filtered[export_cols].to_csv(f"{args.site}_4c_final.csv")
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
