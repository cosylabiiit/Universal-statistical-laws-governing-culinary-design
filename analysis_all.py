import pandas as pd
import os
import matplotlib.pyplot as plt
from nutrient_matrix import NutrientMatrix
from nutrient_plotter import NutrientPlotter
import warnings
warnings.filterwarnings('ignore')

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
print("  ALL RECIPES - NUTRIENT DISTRIBUTION ANALYSIS")
print("=" * 70)

# Load All Recipes (Whole Dataset)
print("\n-- Loading All Recipes Data -----------------------------------------")

all_df = pd.read_csv("DATA/PROCESSED/RecipeDB_Cuisine.csv")
nutrient_cols = ["Carbohydrate, by difference (g)", "Protein (g)", "Total lipid (fat) (g)"]
all_nutrients_df = all_df[nutrient_cols].fillna(0).copy()
print(f"   Total recipes loaded: {len(all_df):,}")

# Initialize Nutrient Analysis Objects
all_nm = NutrientMatrix(all_nutrients_df)
all_plotter = NutrientPlotter(all_nm)

# Create output directory
save_dir = 'WORLD/All'
os.makedirs(save_dir, exist_ok=True)


# Nutrient Prevalence Table
# Shows the proportion of recipes containing each nutrient (p_n > 0)
print("\n-- Nutrient Prevalence Table ----------------------------------")
print(f"   Computing proportion of recipes containing each nutrient ({SYMBOLS['p_n']} > 0)")

all_prevalence = all_nm.prevalence_table()
all_prevalence.to_csv('DATA/PROCESSED/All_Recipes_Nutrient_Prevalence.csv', index=False, float_format='%.4f')
print("   Saved: DATA/PROCESSED/All_Recipes_Nutrient_Prevalence.csv")


# Q(x_n) Distribution for Selected Nutrients
print(f"\n-- Q({SYMBOLS['x_n']}) Distribution Plots ----------------------------------")

selected_nutrients = {
    "Carbohydrate, by difference (g)": {"label": "Carbohydrate", "color": "#d73027"},
    "Protein (g)": {"label": "Protein", "color": "#58ff87"},
    "Total lipid (fat) (g)": {"label": "Fat", "color": "#4575b4"},
}

# Individual nutrient plots
for nname, props in selected_nutrients.items():
    fig, ax = all_plotter.plot_Qxn({nname: props})
    fig.savefig(f"{save_dir}/Qxn_{props['label']}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: Qxn_{props['label']}.png")


# Combined overlay
fig, ax = all_plotter.plot_Qxn(selected_nutrients)
fig.savefig(f"{save_dir}/Qxn_All_Nutrients.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"   Saved: Qxn_All_Nutrients.png")


# Constant Standard Deviation (s_n vs mu_n)
print(f"\n-- Constant Standard Deviation ({SYMBOLS['s_n']} vs {SYMBOLS['mu_n']}) ------------------")

all_stats = all_nm.all_stats()
stats_records = []

for _, row in all_stats.iterrows():
    nutrient = row['nutrient']
    if 'Carbohydrate' in nutrient:
        stats_records.append({
            'Nutrient': 'Carbohydrate',
            'Mean (g)': row['mean'],
            'Log Std (s_n)': row['log_std']
        })
    elif 'Protein' in nutrient:
        stats_records.append({
            'Nutrient': 'Protein',
            'Mean (g)': row['mean'],
            'Log Std (s_n)': row['log_std']
        })
    elif 'lipid' in nutrient or 'fat' in nutrient.lower():
        stats_records.append({
            'Nutrient': 'Fat',
            'Mean (g)': row['mean'],
            'Log Std (s_n)': row['log_std']
        })

all_stats_df = pd.DataFrame(stats_records)
all_stats_df.to_csv('DATA/PROCESSED/All_Recipes_Nutrient_Statistics.csv', index=False, float_format='%.4f')
print("   Saved: DATA/PROCESSED/All_Recipes_Nutrient_Statistics.csv")

fig, ax, summary = all_plotter.plot_log_std()
fig.savefig(f"{save_dir}/Log_Std_vs_Mean.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"   All Recipes: {SYMBOLS['s_n']} = {summary['mean']:.4f} {SYMBOLS['plus_minus']} {summary['std']:.4f}")

fig, ax = all_plotter.plot_raw_std()
fig.savefig(f"{save_dir}/Raw_Std_vs_Mean.png", dpi=300, bbox_inches='tight')
plt.close(fig)


# Symmetry (Logarithmic Skewness)
print("\n-- Symmetry - Logarithmic Skewness ----------------------------")

all_skewness_summary = all_nm.log_skew_summary()

skewness_df = pd.DataFrame([{
    'Cuisine': 'All',
    'mean_sk_n': all_skewness_summary['mean'],
    'Std': all_skewness_summary['std']
}])
skewness_df.to_csv('DATA/PROCESSED/All_Recipes_Log_Skewness.csv', index=False, float_format='%.4f')
print("   Saved: DATA/PROCESSED/All_Recipes_Log_Skewness.csv")

fig, ax, summary = all_plotter.plot_log_skew()
fig.savefig(f"{save_dir}/Log_Skewness_vs_Mean.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"   All Recipes: {SYMBOLS['sk_n']} = {summary['mean']:.4f} {SYMBOLS['plus_minus']} {summary['std']:.4f}")


# Translational Invariance in Log Space
print("\n-- Translational Invariance in Log Space ---------------------")

nutrients_to_rescale = ["Protein (g)", "Total lipid (fat) (g)", "Carbohydrate, by difference (g)"]

fig, ax, records = all_plotter.plot_rescaled_Q(nutrients_to_rescale)
fig.savefig(f"{save_dir}/Rescaled_Q_Translational_Invariance.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"   Saved: Rescaled_Q_Translational_Invariance.png")


# Kolmogorov-Smirnov Test
print("\n-- Kolmogorov-Smirnov Test ------------------------------------")

distributions = ['lognormal', 'gamma', 'weibull', 'gaussian', 'exponential', 'uniform']
print(f"   Testing {len(distributions)} distributions: {', '.join(d.capitalize() for d in distributions)}")

all_ks_df = all_nm.all_ks_tests()

mean_row = {'Cuisine': 'All'}
pct_row = {'Cuisine': 'All'}

for dist in distributions:
    col = f'D_{dist}'
    if col in all_ks_df.columns:
        mean_row[f'{dist.capitalize()} D'] = round(all_ks_df[col].mean(), 4)
        pct_row[f'{dist.capitalize()}'] = round((all_ks_df[col] < 0.1).mean() * 100, 2)
        mean_d = mean_row[f'{dist.capitalize()} D']
        pct_below = pct_row[f'{dist.capitalize()}']
        print(f"   {dist.capitalize():<12}: D = {mean_d:.4f}, % below 0.1 = {pct_below}%")

mean_d_df = pd.DataFrame([mean_row])
pct_below_df = pd.DataFrame([pct_row])

mean_d_df.to_csv('DATA/PROCESSED/All_Recipes_KS_Mean_D.csv', index=False, float_format='%.4f')
pct_below_df.to_csv('DATA/PROCESSED/All_Recipes_KS_Pct_Below_0.1.csv', index=False, float_format='%.4f')
print("   Saved: DATA/PROCESSED/All_Recipes_KS_Mean_D.csv")
print("   Saved: DATA/PROCESSED/All_Recipes_KS_Pct_Below_0.1.csv")


# CDF Comparison Plot
print("\n   Generating CDF comparison plot (Empirical vs Fitted)...")
fig, ax = all_plotter.plot_ks_distance_cdf(all_ks_df)
fig.savefig(f"{save_dir}/KS_Distance_CDF.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"   Saved: KS_Distance_CDF.png")



# sigma_n vs mu_n - Power-Law Scaling
print(f"\n-- {SYMBOLS['sigma_n']} vs {SYMBOLS['mu_n']} - Power-Law Scaling ------------------------------")

fig, ax, params = all_plotter.plot_sigma_vs_mu()
fig.savefig(f"{save_dir}/Sigma_vs_Mu_PowerLaw.png", dpi=300, bbox_inches='tight')
plt.close(fig)

powerlaw_df = pd.DataFrame([{
    'Cuisine': 'All',
    'Slope': round(params['slope'], 4),
    'R2': round(params['R_squared'], 4)
}])
powerlaw_df.to_csv('DATA/PROCESSED/All_Recipes_PowerLaw_Scaling.csv', index=False, float_format='%.4f')
print("   Saved: DATA/PROCESSED/All_Recipes_PowerLaw_Scaling.csv")

print(f"   All Recipes: slope = {params['slope']:.4f}   {SYMBOLS['R2']} = {params['R_squared']:.4f}")
