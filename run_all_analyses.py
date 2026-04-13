import data_preprocessing

def main():    
    data_preprocessing.run()
        
    import analysis_all
    import analysis_top10_cuisine_individually
    import analysis_top10_cuisine_combined
    import analysis_all_cuisines_csv
    analysis_all_cuisines_csv.main()

    print("=" * 70)
    print("\n  Output:")
    print("   WORLD/All/          - All recipes plots")
    print("   WORLD/{cuisine}/    - Individual cuisine plots")
    print("   WORLD/Combined/     - Combined top 10 plots")
    print("   DATA/PROCESSED/     - CSV tables")
    print("   DATA/PROCESSED/ALL/ - All cuisines CSV tables")
    print("=" * 70 + "\n")
    

if __name__ == "__main__":
    main()