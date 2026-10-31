import os
import pandas as pd
from github import Github, Auth

def register_dataset():
    token = os.environ.get("GH_TOKEN")
    if not token:
        print("Error: GH_TOKEN environment variable is not set.")
        return

    # Initialize GitHub client using the recommended Auth.Token class
    auth = Auth.Token(token)
    g = Github(auth=auth)
    repo_name = "amitarathodar-bot/tourism-package-prediction"
    repo = g.get_repo(repo_name)

    try:
        # Fetching content of tourism_project/data/tourism.csv from repo
        file_content = repo.get_contents("tourism_project/data/tourism.csv", ref="main")
        download_url = file_content.download_url
        print(f"Fetching dataset from: {download_url}")

        # Read dataframe directly
        df = pd.read_csv(download_url)
    except Exception as e:
        print(f"Failed to fetch data from repo: {e}")
        return

    # Define expected columns for a typical tourism prediction dataset
    expected_cols = ["Age", "TypeofContact", "CityTier", "DurationOfPitch", "Occupation",
                     "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
                     "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
                     "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
                     "Designation", "MonthlyIncome", "ProdTaken"]

    missing_cols = [col for col in expected_cols if col not in df.columns]

    print("\n--- Verification Report ---")
    if missing_cols:
        print(f"Warning: The following expected columns are missing: {missing_cols}")
    else:
        print("Success: All expected columns are present in the dataset.")

    print("\n--- Dataset Summary ---")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print("Target class distribution ('ProdTaken'):")
    if "ProdTaken" in df.columns:
        print(df["ProdTaken"].value_counts(dropna=False))
    else:
        print("Target column 'ProdTaken' not found.")

if __name__ == '__main__':
    register_dataset()
