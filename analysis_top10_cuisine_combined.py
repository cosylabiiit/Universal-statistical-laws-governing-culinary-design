import pandas as pd
import os
import matplotlib.pyplot as plt
from nutrient_matrix import NutrientMatrix
from combined_plotter import CombinedPlotter
import warnings
warnings.filterwarnings('ignore')

CUISINE_COLORS = {
    "Italian":    "#e41a1c",  # Red
    "Mexican":    "#377eb8",  # Blue
    "South American":  "#ff7f00",  # Orange
    "Canadian":   "#4daf4a",  # Green
    "Indian Subcontinent":     "#ffff33",  # Yellow
    "French":     "#984ea3",  # Purple
    "Chinese and Mongolian":    "#a65628",  # Brown
    "Australian": "#999999",  # Gray
    "US":         "#f781bf",  # Pink
    "UK":      "#66c2a5",  # Teal
}

top10 = [
    "Italian_recipes.csv",
    "Mexican_recipes.csv",
    "South American_recipes.csv",
    "Canadian_recipes.csv",
    "Indian Subcontinent_recipes.csv",
    "French_recipes.csv",
    "Chinese and Mongolian_recipes.csv",
    "Australian_recipes.csv",
    "US_recipes.csv",
    "UK_recipes.csv"
]

SYMBOLS = {
    "x_n": "x\u2099",
    "s_n": "s\u2099",
    "mu_n": "\u03bc\u2099",
    "mean_s_n": "\u27e8s\u2099\u27e9",
    "sk_n": "sk\u2099",
    "sigma_n": "\u03c3\u2099",
    "plus_minus": "\u00b1",
}

print("\n" + "=" * 70)
print("  TOP 10 CUISINES - COMBINED NUTRIENT DISTRIBUTION ANALYSIS")
print("=" * 70)

print("\n-- Loading Top 10 Cuisine Data --------------------------------------")
cuisine_dfs = {}
for fname in top10:
    cuisine_name = fname.replace("_recipes.csv", "")
    df = pd.read_csv(f"DATA/PROCESSED/CUISINES/{fname}")

    nutrient_cols = ["Carbohydrate, by difference (g)", "Protein (g)", "Total lipid (fat) (g)"]
    cuisine_dfs[cuisine_name] = df[nutrient_cols].fillna(0).copy()
    print(f"   {cuisine_name:<15} {len(df):>8,} recipes")

print(f"\n   Total cuisines loaded: {len(cuisine_dfs)}")

# Initialize Nutrient Analysis Objects
# Create NutrientMatrix instances for each cuisine
cuisine_nm = {}
for name, df in cuisine_dfs.items():
    nm = NutrientMatrix(df)
    cuisine_nm[name] = nm

combined_plotter = CombinedPlotter(cuisine_nm)
os.makedirs('WORLD/Combined', exist_ok=True)


# Q(x_n) Distribution - Individual Nutrients Combined
print(f"\n-- Q({SYMBOLS['x_n']}) Distribution - Individual Nutrients -----------------")

for nutrient in CombinedPlotter.NUTRIENTS_LIST:
    fig, ax = combined_plotter.plot_Qxn_individual(nutrient)
    name = CombinedPlotter.NUTRIENT_SHORT[nutrient]
    fig.savefig(f"WORLD/Combined/Qxn_{name}_Combined.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: Qxn_{name}_Combined.png")


# Q(x_n) Distribution - All Nutrients Combined
print(f"\n-- Q({SYMBOLS['x_n']}) Distribution - All Nutrients Combined ---------------")

fig, ax = combined_plotter.plot_Qxn_combined()
fig.savefig("WORLD/Combined/Qxn_All_Nutrients_Combined.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("   Saved: Qxn_All_Nutrients_Combined.png")


# Constant Standard Deviation (s_n vs mu_n)
print(f"\n-- Constant Standard Deviation ({SYMBOLS['s_n']} vs {SYMBOLS['mu_n']}) --------------------")

fig, ax, log_std_metrics, summary = combined_plotter.plot_log_std_combined()
fig.savefig("WORLD/Combined/Log_Std_vs_Mean_Combined.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("   Saved: Log_Std_vs_Mean_Combined.png")
print(f"   {SYMBOLS['mean_s_n']} = {summary['mean']:.4f} {SYMBOLS['plus_minus']} {summary['std']:.4f}")


# Logarithmic Skewness (sk_n vs mu_n)
print(f"\n-- Logarithmic Skewness ({SYMBOLS['sk_n']} vs {SYMBOLS['mu_n']}) --------------------------")

fig, ax, log_skew_metrics = combined_plotter.plot_log_skew_combined()
fig.savefig("WORLD/Combined/Log_Skewness_vs_Mean_Combined.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("   Saved: Log_Skewness_vs_Mean_Combined.png")


# Translational Invariance in Log Space
print("\n-- Translational Invariance in Log Space ----------------------")

rescaled_metrics = []
for nutrient in CombinedPlotter.NUTRIENTS_LIST:
    fig, ax, metrics = combined_plotter.plot_rescaled_Q_individual(nutrient)
    short_name = CombinedPlotter.NUTRIENT_SHORT[nutrient]
    fig.savefig(f"WORLD/Combined/Rescaled_Q_{short_name}_Combined.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    rescaled_metrics.extend(metrics)
    print(f"   Saved: Rescaled_Q_{short_name}_Combined.png")


# sigma_n vs mu_n - Power-Law Scaling
print(f"\n-- {SYMBOLS['sigma_n']} vs {SYMBOLS['mu_n']} - Power-Law Scaling -------------------------------")

fig, ax, powerlaw_metrics = combined_plotter.plot_powerlaw_combined()
fig.savefig("WORLD/Combined/Sigma_vs_Mu_PowerLaw_Combined.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("   Saved: Sigma_vs_Mu_PowerLaw_Combined.png")


# K-S Distance CDF - Best Distribution by Cuisine
print("\n-- K-S Distance CDF - Best Distribution by Cuisine ------------")

fig, ax, best_dist_df = combined_plotter.plot_ks_distance_cdf_combined()
fig.savefig("WORLD/Combined/KS_Distance_CDF_Best_Distribution_Combined.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("   Saved: KS_Distance_CDF_Best_Distribution_Combined.png")


# Save Combined Metrics to CSV
print("\n-- Saving Combined Metrics to CSV ------------------------------------")

saved_files = combined_plotter.save_metrics_to_csv(
    log_std_metrics=log_std_metrics,
    log_skew_metrics=log_skew_metrics,
    powerlaw_metrics=powerlaw_metrics,
    rescaled_metrics=rescaled_metrics,
    output_dir='DATA/PROCESSED'
)

for key, path in saved_files.items():
    print(f"   {key}: {path}")
