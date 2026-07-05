iimport matplotlib.pyplot as plt

def plot_4c_grid(df, channels, outfile, plot_func, title_base, site_name):
    """General helper to create 2x2 grids for TESS-4C"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, ch in enumerate(channels):
        # We pass a temporary subset df to the plotting function
        # Note: You may need to adapt your existing plot functions 
        # to accept an 'ax' argument for these grids
        plot_func(df, ch, axes[i], site_name=f"{site_name} ({ch})")
        
    plt.suptitle(f"{title_base} ({site_name})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(outfile, dpi=150)
    plt.close()

# Add specific 4C-compliant plotting functions here that accept an 'ax' argument
