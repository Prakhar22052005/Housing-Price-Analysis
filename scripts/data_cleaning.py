"""
scripts/data_cleaning.py
========================
Cross-Platform Data Cleaning and Feature Preprocessing Pipeline for Ames Housing Dataset.
Compatible with Google Colab, Linux, macOS, and Windows.

This script executes:
1. Ingesting raw Ames Housing data (with automated public download fallback if missing).
2. Programmatic quality audit (dimensions, column structure, missing values).
3. Missing value imputation differentiating structural absence ('None' / 0) from missing observations.
4. Conditional outlier handling: Flags severe leverage outliers (GrLivArea > 4000 & SalePrice < $300k)
   while preserving legitimate luxury properties.
5. Domain feature engineering (Total_SF, Property_Age, Remodel_Age, Total_Bathrooms, Bath_to_Bed_Ratio, etc.).
6. Export of cleaned dataset to data/processed/Ames_Housing_Cleaned.csv.

Note: In accordance with data leakage prevention principles, 'Price_per_SF' is EXCLUDED
from the cleaned modeling dataset and calculated strictly during descriptive EDA.
"""

import os
import sys
import argparse
import urllib.request
import pandas as pd
import numpy as np

PUBLIC_RAW_DATA_URL = "https://raw.githubusercontent.com/ds4stats/r-tutorials/master/data-viz/data/AmesHousing.csv"


def ensure_raw_dataset(raw_path: str) -> str:
    """Ensures raw dataset exists, downloading from public repository if absent."""
    if os.path.exists(raw_path):
        return raw_path

    print(f"[INFO] Raw dataset not found at '{raw_path}'.")
    print(f"[INFO] Downloading official Ames Housing dataset from: {PUBLIC_RAW_DATA_URL} ...")
    os.makedirs(os.path.dirname(os.path.abspath(raw_path)), exist_ok=True)
    try:
        req = urllib.request.Request(PUBLIC_RAW_DATA_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response, open(raw_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"[SUCCESS] Downloaded and saved raw dataset to: {raw_path}")
        return raw_path
    except Exception as e:
        raise RuntimeError(f"Failed to automatically download dataset: {e}. Please place Ames_Housing_Raw.csv at {raw_path}")


def clean_ames_housing(raw_path: str, output_path: str) -> pd.DataFrame:
    raw_path = os.path.abspath(raw_path)
    output_path = os.path.abspath(output_path)

    ensure_raw_dataset(raw_path)
    print(f"[INFO] Ingesting raw dataset from: {raw_path}")
    df = pd.read_csv(raw_path)
    initial_shape = df.shape
    print(f"[INFO] Initial shape: {initial_shape[0]} rows x {initial_shape[1]} columns")

    # 1. Standardize column names (remove slashes, spaces)
    df = df.rename(columns={
        'YearRemod/Add': 'YearRemodAdd',
        'MS SubClass': 'MSSubClass',
        'MS Zoning': 'MSZoning'
    })

    # Preserve administrative columns for tracking
    admin_cols = [c for c in ['Order', 'PID'] if c in df.columns]
    print(f"[INFO] Administrative identifier columns: {admin_cols}")

    # 2. Check duplicate records
    dup_count = df.duplicated(subset=['PID'] if 'PID' in df.columns else None).sum()
    print(f"[INFO] Duplicate records identified: {dup_count}")

    # 3. Conditional Outlier Handling
    # De Cock (2011) recommends removing 3 extreme outliers with GrLivArea > 4000 sq ft that sold for unusually low prices (< $300k)
    # while retaining legitimate high-end luxury homes (> 4000 sq ft and high SalePrice).
    extreme_outliers = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index
    print(f"[INFO] Removing {len(extreme_outliers)} severe leverage outlier(s) with GrLivArea > 4000 and SalePrice < $300,000.")
    print(f"       Preserved legitimate luxury properties > 4,000 sq ft with corresponding market value.")
    df = df.drop(index=extreme_outliers).reset_index(drop=True)

    # 4. Handle Missing Values based on Ames Documentation
    # A) Categorical features where NA means absence of amenity
    categorical_none_cols = [
        'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType'
    ]
    for col in categorical_none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # B) Numerical features where NA corresponds to absence (count or area is 0)
    numeric_zero_cols = [
        'GarageYrBlt', 'GarageCars', 'GarageArea', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in numeric_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # C) True missing observations: LotFrontage (impute by Neighborhood median)
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

    # D) Single/rare missing categorical values: mode imputation
    for col in ['Electrical', 'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 5. Feature Engineering (Domain specific)
    # Total square footage (Basement + 1st Floor + 2nd Floor)
    df['Total_SF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # Property age at time of sale
    df['Property_Age'] = (df['YrSold'] - df['YearBuilt']).clip(lower=0)

    # Remodel age at time of sale
    df['Remodel_Age'] = (df['YrSold'] - df['YearRemodAdd']).clip(lower=0)

    # Remodel binary indicator
    df['Is_Remodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)

    # Total bathrooms (Full + 0.5 * Half)
    df['Total_Bathrooms'] = (
        df['FullBath'] + 0.5 * df['HalfBath'] +
        df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    )

    # Bath to bedroom ratio
    df['Bath_to_Bed_Ratio'] = df['Total_Bathrooms'] / df['BedroomAbvGr'].replace(0, 1)

    # Combined outdoor porch/deck square footage
    porch_cols = [c for c in ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'] if c in df.columns]
    df['Outdoor_Porch_SF'] = df[porch_cols].sum(axis=1)

    # Verify no null values remain
    null_remaining = df.isnull().sum().sum()
    print(f"[INFO] Missing values remaining across all columns: {null_remaining}")

    # Output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Cleaned dataset saved to: {output_path}")
    print(f"[INFO] Final cleaned shape: {df.shape[0]} rows x {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    parser = argparse.ArgumentParser(description="Clean Ames Housing Dataset")
    parser.add_argument(
        "--raw_path",
        type=str,
        default=os.path.join(project_root, "data", "raw", "Ames_Housing_Raw.csv"),
        help="Path to raw CSV dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(project_root, "data", "processed", "Ames_Housing_Cleaned.csv"),
        help="Path to save cleaned CSV"
    )
    args = parser.parse_args()
    clean_ames_housing(args.raw_path, args.output_path)
