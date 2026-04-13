import pandas as pd
import os
import matplotlib.pyplot as plt
from nutrient_matrix import NutrientMatrix
from nutrient_plotter import NutrientPlotter
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
    "p_n": "p\u2099",
    "x_n": "x\u2099",
    "s_n": "s\u2099",
    "mu_n": "\u03bc\u2099",
    "sk_n": "sk\u2099",
    "sigma_n": "\u03c3\u2099",
    "plus_minus": "\u00b1",
    "R2": "R\u00b2",
}

print("\n" + "=" * 70)
print("  TOP 10 CUISINES - NUTRIENT DISTRIBUTION ANALYSIS")
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
# Create NutrientMatrix and NutrientPlotter instances for each cuisine
cuisine_nm = {}
cuisine_plotter = {}

for name, df in cuisine_dfs.items():
    nm = NutrientMatrix(df)
    cuisine_nm[name] = nm
    cuisine_plotter[name] = NutrientPlotter(nm)

os.makedirs('WORLD', exist_ok=True)
for cuisine in cuisine_nm.keys():
    os.makedirs(f'WORLD/{cuisine}', exist_ok=True)


# Nutrient Prevalence Table
# Shows the proportion of recipes containing each nutrient (p_n > 0)
print("\n-- Nutrient Prevalence Table ----------------------------------")
print(f"   Computing proportion of recipes containing each nutrient ({SYMBOLS['p_n']} > 0)")

prevalence_records = []
for name, nm in cuisine_nm.items():
    prev_table = nm.prevalence_table()
    row_data = {'cuisine': name}
    for _, row in prev_table.iterrows():
        row_data[row['Nutrient']] = row['p_n']
    prevalence_records.append(row_data)

prevalence_df = pd.DataFrame(prevalence_records)
prevalence_df = prevalence_df[['cuisine', 'Total lipid (fat) (g)', 'Protein (g)', 'Carbohydrate, by difference (g)']]
prevalence_df.to_csv('DATA/PROCESSED/Cuisine_Nutrient_Prevalence.csv', index=False, float_format='%.4f')
print("   Saved: DATA/PROCESSED/Cuisine_Nutrient_Prevalence.csv")


# Q(x_n) Distribution for Selected Nutrients
print(f"\n-- Q({SYMBOLS['x_n']}) Distribution Plots ----------------------------------")

selected_nutrients = {
    "Carbohydrate, by difference (g)": {"label": "Carbohydrate", "color": "#d73027"},
    "Protein (g)": {"label": "Protein", "color": "#58ff87"},
    "Total lipid (fat) (g)": {"label": "Fat", "color": "#4575b4"},
}

for name, plotter in cuisine_plotter.items():
    save_dir = f'WORLD/{name}'
    
    # Individual nutrient plots
    for nname, props in selected_nutrients.items():
        fig, ax = plotter.plot_Qxn({nname: props})
        fig.savefig(f"{save_dir}/Qxn_{props['label']}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Combined overlay
    fig, ax = plotter.plot_Qxn(selected_nutrients)
    fig.savefig(f"{save_dir}/Qxn_All_Nutrients.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   {name:<15} -> 4 Q({SYMBOLS['x_n']}) plots saved")

print(f"   All Q({SYMBOLS['x_n']}) plots saved to WORLD/{{cuisine}}/")


# Constant Standard Deviation (s_n vs mu_n)
print(f"\n-- Constant Standard Deviation ({SYMBOLS['s_n']} vs {SYMBOLS['mu_n']}) ------------------")

nutrient_stats_records = []

for name, nm in cuisine_nm.items():
    stats = nm.all_stats()
    row_data = {'Cuisine': name}
    
    for _, row in stats.iterrows():
        nutrient = row['nutrient']
        if 'Carbohydrate' in nutrient:
            row_data['Carbohydrate mu_n (g)'] = row['mean']
            row_data['Carbohydrate s_n'] = row['log_std']
        elif 'Protein' in nutrient:
            row_data['Protein mu_n (g)'] = row['mean']
            row_data['Protein s_n'] = row['log_std']
        elif 'lipid' in nutrient or 'fat' in nutrient.lower():
            row_data['Fat mu_n (g)'] = row['mean']
            row_data['Fat s_n'] = row['log_std']
    
    nutrient_stats_records.append(row_data)

nutrient_stats_df = pd.DataFrame(nutrient_stats_records)
nutrient_stats_df = nutrient_stats_df[['Cuisine', 'Carbohydrate mu_n (g)', 'Carbohydrate s_n', 'Protein mu_n (g)', 'Protein s_n', 'Fat mu_n (g)', 'Fat s_n']]
nutrient_stats_df.to_csv('DATA/PROCESSED/Cuisine_Nutrient_Statistics.csv', index=False, float_format='%.4f')
print("   Saved: DATA/PROCESSED/Cuisine_Nutrient_Statistics.csv")

for name, plotter in cuisine_plotter.items():
    fig, ax, summary = plotter.plot_log_std()
    save_dir = f'WORLD/{name}'
    fig.savefig(f"{save_dir}/Log_Std_vs_Mean.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   {name:<15} {SYMBOLS['s_n']} = {summary['mean']:.4f} {SYMBOLS['plus_minus']} {summary['std']:.4f}")


# Symmetry (Logarithmic Skewness)
print("\n-- Symmetry - Logarithmic Skewness ----------------------------")

skewness_records = []

for name, nm in cuisine_nm.items():
    summary = nm.log_skew_summary()
    skewness_records.append({
        'Cuisine': name,
        'mean_sk_n': summary['mean'],
        'Std': summary['std']
    })

skewness_df = pd.DataFrame(skewness_records)
skewness_df.to_csv('DATA/PROCESSED/Cuisine_Log_Skewness.csv', index=False, float_format='%.4f')
print("   Saved: DATA/PROCESSED/Cuisine_Log_Skewness.csv")

for name, plotter in cuisine_plotter.items():
    fig, ax, summary = plotter.plot_log_skew()
    save_dir = f'WORLD/{name}'
    fig.savefig(f"{save_dir}/Log_Skewness_vs_Mean.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   {name:<15} {SYMBOLS['sk_n']} = {summary['mean']:.4f} {SYMBOLS['plus_minus']} {summary['std']:.4f}")


# Translational Invariance in Log Space
print("\n-- Translational Invariance in Log Space ---------------------")

nutrients_to_rescale = ["Protein (g)", "Total lipid (fat) (g)", "Carbohydrate, by difference (g)"]

for name, plotter in cuisine_plotter.items():
    fig, ax, records = plotter.plot_rescaled_Q(nutrients_to_rescale)
    save_dir = f'WORLD/{name}'
    fig.savefig(f"{save_dir}/Rescaled_Q_Translational_Invariance.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   {name:<15} -> Rescaled Q plot saved")


# Kolmogorov-Smirnov Test
print("\n-- Kolmogorov-Smirnov Test ------------------------------------")

distributions = ['lognormal', 'gamma', 'weibull', 'gaussian', 'exponential', 'uniform']
print(f"   Testing {len(distributions)} distributions: {', '.join(d.capitalize() for d in distributions)}")

mean_d_records = []
pct_below_records = []

for name, nm in cuisine_nm.items():
    ks_df = nm.all_ks_tests()
    
    mean_row = {'Cuisine': name}
    pct_row = {'Cuisine': name}
    
    for dist in distributions:
        col = f'D_{dist}'
        if col in ks_df.columns:
            mean_row[f'{dist.capitalize()} D'] = round(ks_df[col].mean(), 4)
            pct_row[f'{dist.capitalize()}'] = round((ks_df[col] < 0.1).mean() * 100, 2)
    
    mean_d_records.append(mean_row)
    pct_below_records.append(pct_row)
    print(f"   {name:<15} -> KS tests complete")

mean_d_df = pd.DataFrame(mean_d_records)
pct_below_df = pd.DataFrame(pct_below_records)

mean_d_df.to_csv('DATA/PROCESSED/Cuisine_KS_Mean_D.csv', index=False, float_format='%.4f')
pct_below_df.to_csv('DATA/PROCESSED/Cuisine_KS_Pct_Below_0.1.csv', index=False, float_format='%.4f')
print("   Saved: DATA/PROCESSED/Cuisine_KS_Mean_D.csv")
print("   Saved: DATA/PROCESSED/Cuisine_KS_Pct_Below_0.1.csv")


# CDF Comparison - Empirical vs Fitted Distributions
print("\n   Generating CDF comparison plots (Empirical vs Fitted)...")
for name, plotter in cuisine_plotter.items():
    ks_df = cuisine_nm[name].all_ks_tests()
    fig, ax = plotter.plot_ks_distance_cdf(ks_df)
    save_dir = f'WORLD/{name}'
    fig.savefig(f"{save_dir}/KS_Distance_CDF.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   {name:<15} -> KS_Distance_CDF.png saved")


# sigma_n vs mu_n - Power-Law Scaling
print(f"\n-- {SYMBOLS['sigma_n']} vs {SYMBOLS['mu_n']} - Power-Law Scaling ------------------------------")

powerlaw_records = []

for name, plotter in cuisine_plotter.items():
    fig, ax, params = plotter.plot_sigma_vs_mu()
    save_dir = f'WORLD/{name}'
    fig.savefig(f"{save_dir}/Sigma_vs_Mu_PowerLaw.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    powerlaw_records.append({
        'Cuisine': name,
        'Slope': round(params['slope'], 4),
        'R2': round(params['R_squared'], 4)
    })
    print(f"   {name:<15} slope = {params['slope']:.4f}   {SYMBOLS['R2']} = {params['R_squared']:.4f}")

powerlaw_df = pd.DataFrame(powerlaw_records)
powerlaw_df.to_csv('DATA/PROCESSED/Cuisine_PowerLaw_Scaling.csv', index=False, float_format='%.4f')
print("   Saved: DATA/PROCESSED/Cuisine_PowerLaw_Scaling.csv")