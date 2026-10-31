import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data():
    print("Starting data preparation with data cleaning...")
    dataset_path = "tourism_project/data/tourism.csv"

    # 1. Load the dataset directly from the repository data folder
    df = pd.read_csv(dataset_path)

    # 2. Perform data cleaning & remove unnecessary columns
    # Drop any row identifier columns if they exist (e.g., 'CustomerID')
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])

    # Handle missing numerical values by filling with the median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Handle missing categorical values by filling with the mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"Data cleaned. Remaining null values: {df.isnull().sum().sum()}")

    # 3. Split the cleaned dataset into training and testing sets (80/20 split)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 4. Save them locally
    os.makedirs("tourism_project/data", exist_ok=True)
    train_df.to_csv("tourism_project/data/train.csv", index=False)
    test_df.to_csv("tourism_project/data/test.csv", index=False)
    print("Data preparation complete! Train and Test splits saved successfully.")

if __name__ == '__main__':
    prepare_data()
