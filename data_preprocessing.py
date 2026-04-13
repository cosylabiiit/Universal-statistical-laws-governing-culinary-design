import pandas as pd
import os


def run():
    print("\n" + "=" * 70)
    print("  DATA PREPROCESSING")
    print("=" * 70)

    os.makedirs('DATA/PROCESSED', exist_ok=True)

    print("\n── Loading Raw Data ─────────────────────────────────────────────")
    df = pd.read_csv('DATA/RAW/RecipeDB_Cuisine_Nutrition.csv')
    df['cuisine'] = df['Region']
    print(f"   Loaded {len(df):,} recipes with {df['cuisine'].nunique()} unique cuisines")
    df.to_csv("DATA/PROCESSED/RecipeDB_Cuisine.csv", index=False, float_format='%.4f')
    print("   Saved: DATA/PROCESSED/RecipeDB_Cuisine.csv")

    print("\n── Splitting Into Per-Cuisine Files ─────────────────────────────")
    cuisines = df['cuisine'].dropna().unique()

    base_path = 'DATA/PROCESSED/CUISINES'
    os.makedirs(base_path, exist_ok=True)

    for cuisine in cuisines:
        cuisine_df = df[df['cuisine'] == cuisine]
        cuisine_df.to_csv(os.path.join(base_path, f'{cuisine_df.iloc[0]["cuisine"]}_recipes.csv'), index=False, float_format='%.4f')

    print(f"   Created {len(cuisines)} cuisine files in {base_path}/")
    print(f"   Cuisines: {', '.join(sorted(cuisines))}")

    print("\n── Top 10 Cuisines Summary ──────────────────────────────────────")
    top10 = df['cuisine'].value_counts().head(10).index.tolist()
    df_top10 = df[df['cuisine'].isin(top10)].copy()

    print(f"   {len(df_top10):,} recipes out of {len(df):,} total ({len(df_top10)/len(df)*100:.1f}% coverage)")
    for cuisine_name in top10:
        count = len(df_top10[df_top10['cuisine'] == cuisine_name])
        print(f"     {cuisine_name:<15} {count:>6,} recipes")

    print("\n── Saving Recipe Counts ─────────────────────────────────────────")
    cuisine_counts = df['cuisine'].value_counts().reset_index()
    cuisine_counts.columns = ['cuisine', 'recipe_count']
    cuisine_counts.to_csv('DATA/PROCESSED/Cuisine_Recipe_Counts.csv', index=False, float_format='%.4f')
    print("   Saved: DATA/PROCESSED/Cuisine_Recipe_Counts.csv")

    print("\n" + "-" * 70)
    print("  Preprocessing complete. All outputs in DATA/PROCESSED/")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    run()
