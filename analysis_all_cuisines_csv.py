import os
import warnings
import pandas as pd
from nutrient_matrix import NutrientMatrix
warnings.filterwarnings("ignore")

NUTRIENT_COLS = ["Carbohydrate, by difference (g)", "Protein (g)", "Total lipid (fat) (g)"]
DISTRIBUTIONS = ["lognormal", "gamma", "weibull", "gaussian", "exponential", "uniform"]

CUISINE_COUNTS_PATH = "DATA/PROCESSED/Cuisine_Recipe_Counts.csv"
CUISINE_DATA_DIR = "DATA/PROCESSED/CUISINES"
OUTPUT_DIR = "DATA/PROCESSED/ALL"

def load_cuisine_names():
    if os.path.exists(CUISINE_COUNTS_PATH):
        counts_df = pd.read_csv(CUISINE_COUNTS_PATH)
        if "cuisine" in counts_df.columns:
            return counts_df["cuisine"].dropna().astype(str).tolist()

    files = [file_name for file_name in os.listdir(CUISINE_DATA_DIR) if file_name.endswith("_recipes.csv")]
    return sorted(file_name.replace("_recipes.csv", "") for file_name in files)


def load_cuisine_data(cuisine_names):
    cuisine_dfs = {}

    print("\n-- Loading all cuisine nutrient data --")
    for cuisine in cuisine_names:
        cuisine_file = os.path.join(CUISINE_DATA_DIR, f"{cuisine}_recipes.csv")

        df = pd.read_csv(cuisine_file)
        cuisine_dfs[cuisine] = df[NUTRIENT_COLS].fillna(0).copy()
        print(f"   {cuisine:<20} {len(df):>8,} recipes")

    print(f"\n   Loaded cuisines: {len(cuisine_dfs)}")
    return cuisine_dfs


def build_nutrient_matrices(cuisine_dfs):
    return {name: NutrientMatrix(df) for name, df in cuisine_dfs.items()}


def ensure_columns(df, ordered_columns):
    for col in ordered_columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[ordered_columns]


def save_prevalence(cuisine_nm):
    print("\n-- Nutrient prevalence --")

    records = []
    for cuisine, nm in cuisine_nm.items():
        prev_table = nm.prevalence_table()
        row_data = {"cuisine": cuisine}

        for _, row in prev_table.iterrows():
            row_data[row["Nutrient"]] = row["p_n"]

        records.append(row_data)

    prevalence_df = pd.DataFrame(records)
    prevalence_df = ensure_columns(
        prevalence_df,
        [
            "cuisine",
            "Total lipid (fat) (g)",
            "Protein (g)",
            "Carbohydrate, by difference (g)",
        ],
    )
    out_path = os.path.join(OUTPUT_DIR, "Cuisine_Nutrient_Prevalence.csv")
    prevalence_df.to_csv(out_path, index=False, float_format='%.4f')
    print(f"   Saved: {out_path}")


def save_nutrient_statistics(cuisine_nm):
    print("\n-- Nutrient statistics --")

    records = []
    for cuisine, nm in cuisine_nm.items():
        stats = nm.all_stats()
        row_data = {
            "Cuisine": cuisine,
            "Carbohydrate mu_n (g)": pd.NA,
            "Carbohydrate s_n": pd.NA,
            "Protein mu_n (g)": pd.NA,
            "Protein s_n": pd.NA,
            "Fat mu_n (g)": pd.NA,
            "Fat s_n": pd.NA,
        }

        for _, row in stats.iterrows():
            nutrient = row["nutrient"]
            if "Carbohydrate" in nutrient:
                row_data["Carbohydrate mu_n (g)"] = row["mean"]
                row_data["Carbohydrate s_n"] = row["log_std"]
            elif "Protein" in nutrient:
                row_data["Protein mu_n (g)"] = row["mean"]
                row_data["Protein s_n"] = row["log_std"]
            elif "lipid" in nutrient or "fat" in nutrient.lower():
                row_data["Fat mu_n (g)"] = row["mean"]
                row_data["Fat s_n"] = row["log_std"]

        records.append(row_data)

    stats_df = pd.DataFrame(records)
    stats_df = ensure_columns(
        stats_df,
        [
            "Cuisine",
            "Carbohydrate mu_n (g)",
            "Carbohydrate s_n",
            "Protein mu_n (g)",
            "Protein s_n",
            "Fat mu_n (g)",
            "Fat s_n",
        ],
    )
    out_path = os.path.join(OUTPUT_DIR, "Cuisine_Nutrient_Statistics.csv")
    stats_df.to_csv(out_path, index=False, float_format='%.4f')
    print(f"   Saved: {out_path}")


def save_log_skewness(cuisine_nm):
    print("\n-- Log-skewness summary --")

    records = []
    for cuisine, nm in cuisine_nm.items():
        summary = nm.log_skew_summary()
        records.append(
            {
                "Cuisine": cuisine,
                "mean_sk_n": summary["mean"],
                "Std": summary["std"],
            }
        )

    skew_df = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_DIR, "Cuisine_Log_Skewness.csv")
    skew_df.to_csv(out_path, index=False, float_format='%.4f')
    print(f"   Saved: {out_path}")


def save_ks_tables(cuisine_nm):
    print("\n-- K-S summary tables --")

    mean_d_records = []
    pct_below_records = []

    for cuisine, nm in cuisine_nm.items():
        ks_df = nm.all_ks_tests()

        mean_row = {"Cuisine": cuisine}
        pct_row = {"Cuisine": cuisine}

        for dist in DISTRIBUTIONS:
            col = f"D_{dist}"
            if col in ks_df.columns:
                mean_row[f"{dist.capitalize()} D"] = round(ks_df[col].mean(), 4)
                pct_row[f"{dist.capitalize()}"] = round((ks_df[col] < 0.1).mean() * 100, 2)

        mean_d_records.append(mean_row)
        pct_below_records.append(pct_row)

    mean_d_df = pd.DataFrame(mean_d_records)
    pct_below_df = pd.DataFrame(pct_below_records)

    mean_d_df = ensure_columns(
        mean_d_df,
        ["Cuisine"] + [f"{dist.capitalize()} D" for dist in DISTRIBUTIONS],
    )
    pct_below_df = ensure_columns(
        pct_below_df,
        ["Cuisine"] + [dist.capitalize() for dist in DISTRIBUTIONS],
    )

    mean_out = os.path.join(OUTPUT_DIR, "Cuisine_KS_Mean_D.csv")
    pct_out = os.path.join(OUTPUT_DIR, "Cuisine_KS_Pct_Below_0.1.csv")

    mean_d_df.to_csv(mean_out, index=False, float_format='%.4f')
    pct_below_df.to_csv(pct_out, index=False, float_format='%.4f')

    print(f"   Saved: {mean_out}")
    print(f"   Saved: {pct_out}")


def save_powerlaw(cuisine_nm):
    print("\n-- Power-law scaling --")

    records = []
    for cuisine, nm in cuisine_nm.items():
        params = nm.linear_fit()
        records.append(
            {
                "Cuisine": cuisine,
                "Slope": round(params["slope"], 4),
                "R2": round(params["R_squared"], 4),
            }
        )

    powerlaw_df = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_DIR, "Cuisine_PowerLaw_Scaling.csv")
    powerlaw_df.to_csv(out_path, index=False, float_format='%.4f')
    print(f"   Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cuisine_names = load_cuisine_names()
    cuisine_dfs = load_cuisine_data(cuisine_names)
    cuisine_nm = build_nutrient_matrices(cuisine_dfs)

    save_prevalence(cuisine_nm)
    save_nutrient_statistics(cuisine_nm)
    save_log_skewness(cuisine_nm)
    save_ks_tables(cuisine_nm)
    save_powerlaw(cuisine_nm)

    print()
    print(f"All CSV files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
