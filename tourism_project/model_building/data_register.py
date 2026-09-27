
import pandas as pd
import os

DATA_PATH = "tourism_project/data/tourism.csv"

def register_dataset(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please upload tourism.csv.")

    df = pd.read_csv(data_path)

    # Define expected columns based on your Data Description
    expected_columns = [
        "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
        "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar",
        "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
        "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
        "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
        "DurationOfPitch"
    ]

    # Check for missing columns
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    print(f"Dataset '{os.path.basename(data_path)}' loaded successfully.")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("First 5 rows:\n", df.head())
    print("Summary statistics:\n", df.describe())
    print("Data types:\n", df.info())

if __name__ == "__main__":
    register_dataset(DATA_PATH)
