
import pandas as pd
from sklearn.model_selection import train_test_split
import os

DATA_PATH = "tourism_project/data/tourism.csv"
OUTPUT_DIR = "."

def prepare_data(data_path, output_dir):
    df = pd.read_csv(data_path)

    # Drop 'Unnamed: 0' column if it exists, as it seems to be an index column
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Handle missing values as identified in initial data exploration
    # For simplicity, filling with median for numerical columns 'MonthlyIncome' and 'Age'
    if 'MonthlyIncome' in df.columns:
        df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    if 'Age' in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)

    # Define features (X) and target (y)
    X = df.drop('ProdTaken', axis=1)
    y = df['ProdTaken']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Save splits locally
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, 'Xtrain.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'Xtest.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'ytrain.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'ytest.csv'), index=False)

    print("Data preparation complete. Train/test splits saved:")
    print(f" - {os.path.join(output_dir, 'Xtrain.csv')}")
    print(f" - {os.path.join(output_dir, 'Xtest.csv')}")
    print(f" - {os.path.join(output_dir, 'ytrain.csv')}")
    print(f" - {os.path.join(output_dir, 'ytest.csv')}")

if __name__ == "__main__":
    prepare_data(DATA_PATH, OUTPUT_DIR)
