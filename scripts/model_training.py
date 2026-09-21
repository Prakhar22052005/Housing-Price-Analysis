"""
scripts/model_training.py
=========================
Cross-Platform Regression Modeling and Evaluation Pipeline for Ames Housing Valuation.
Compatible with Google Colab, Linux, macOS, and Windows.

Features:
1. Ingests data/processed/Ames_Housing_Cleaned.csv.
2. Enforces data leakage prevention:
   - Strict 80/20 train/test split before fitting any transformers.
   - Strictly EXCLUDES Price_per_SF from feature matrix.
   - Drops administrative ID columns (Order, PID).
3. Constructs scikit-learn ColumnTransformer for numerical scaling and one-hot encoding.
4. Trains 6 benchmark models:
   - 1. Baseline (Dummy Regressor - Median)
   - 2. Linear Regression (OLS)
   - 3. Ridge Regression (L2 Regularization)
   - 4. Decision Tree Regressor
   - 5. Random Forest Regressor
   - 6. Gradient Boosting Regressor
5. Computes and tabulates evaluation metrics: MAE, RMSE, R² on held-out test data.
6. Generates and exports evaluation charts to outputs/charts/:
   - 06_model_performance_comparison.png
   - 07_feature_importance.png
   - 08_residuals_analysis.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_and_evaluate_models(data_path: str, charts_dir: str):
    data_path = os.path.abspath(data_path)
    charts_dir = os.path.abspath(charts_dir)

    print(f"[INFO] Ingesting cleaned data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found: {data_path}. Run data_cleaning.py first.")

    df = pd.read_csv(data_path)
    os.makedirs(charts_dir, exist_ok=True)

    # 1. Target and Feature Separation
    target_col = 'SalePrice'
    drop_cols = ['Order', 'PID', target_col]
    if 'Price_per_SF' in df.columns:
        drop_cols.append('Price_per_SF')

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df[target_col]

    print(f"[INFO] Modeling Matrix: {X.shape[0]} samples, {X.shape[1]} input features.")
    print(f"[LEAKAGE CHECK] 'Price_per_SF' in features: {'Price_per_SF' in X.columns} (Must be False)")

    # 2. Identify numerical vs categorical column types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    print(f"[INFO] Categorical features: {len(cat_cols)}, Numerical features: {len(num_cols)}")

    # 3. Strict Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[INFO] Train split: {X_train.shape[0]} samples, Test split: {X_test.shape[0]} samples.")

    # 4. Pipeline Preprocessing (fit strictly on training fold)
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    # Fit preprocessor only on train fold
    preprocessor.fit(X_train)
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    print(f"[INFO] Transformed feature space: {X_train_trans.shape[1]} columns after encoding.")

    # 5. Define Model Lineup
    models = {
        "Baseline (Median)": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    }

    results = []
    trained_models = {}
    test_preds = {}

    print("\n" + "=" * 75)
    print(f"{'Model':<22} | {'MAE ($)':<12} | {'RMSE ($)':<12} | {'R² Score':<10}")
    print("=" * 75)

    for name, model in models.items():
        model.fit(X_train_trans, y_train)
        preds = model.predict(X_test_trans)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })
        trained_models[name] = model
        test_preds[name] = preds

        print(f"{name:<22} | ${mae:11,.2f} | ${rmse:11,.2f} | {r2:10.4f}")

    print("=" * 75 + "\n")
    results_df = pd.DataFrame(results)

    # 6. Visualization 1: Model Benchmark Comparison
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # R2 Comparison
    sns.barplot(data=results_df, x='Model', y='R2', ax=axes[0], palette='Blues_d', hue='Model', legend=False)
    axes[0].set_title('Model Goodness-of-Fit (R² Score)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('R² Score', fontsize=11)
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis='x', rotation=30)
    for p in axes[0].patches:
        val = p.get_height()
        if val > 0:
            axes[0].annotate(f"{val:.3f}", (p.get_x() + p.get_width() / 2., val),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    # MAE Comparison
    sns.barplot(data=results_df, x='Model', y='MAE', ax=axes[1], palette='Oranges_d', hue='Model', legend=False)
    axes[1].set_title('Mean Absolute Error (MAE)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('MAE ($)', fontsize=11)
    axes[1].tick_params(axis='x', rotation=30)
    for p in axes[1].patches:
        val = p.get_height()
        axes[1].annotate(f"${val:,.0f}", (p.get_x() + p.get_width() / 2., val),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    # RMSE Comparison
    sns.barplot(data=results_df, x='Model', y='RMSE', ax=axes[2], palette='Reds_d', hue='Model', legend=False)
    axes[2].set_title('Root Mean Squared Error (RMSE)', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('RMSE ($)', fontsize=11)
    axes[2].tick_params(axis='x', rotation=30)
    for p in axes[2].patches:
        val = p.get_height()
        axes[2].annotate(f"${val:,.0f}", (p.get_x() + p.get_width() / 2., val),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()
    chart1_path = os.path.join(charts_dir, '06_model_performance_comparison.png')
    fig.savefig(chart1_path, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Saved model performance comparison to: {chart1_path}")

    # 7. Visualization 2: Feature Importance (Gradient Boosting)
    gb_model = trained_models["Gradient Boosting"]
    importances = gb_model.feature_importances_
    cleaned_names = [fn.replace('num__', '').replace('cat__', '') for fn in feature_names]
    feat_df = pd.DataFrame({'Feature': cleaned_names, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_df, x='Importance', y='Feature', ax=ax, palette='viridis', hue='Feature', legend=False)
    ax.set_title('Top 15 Predictive Features (Gradient Boosting)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Relative Importance (Gini Impurity Reduction)', fontsize=11)
    ax.set_ylabel('Feature Name', fontsize=11)
    plt.tight_layout()
    chart2_path = os.path.join(charts_dir, '07_feature_importance.png')
    fig.savefig(chart2_path, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Saved feature importance chart to: {chart2_path}")

    # 8. Visualization 3: Residual Analysis (Gradient Boosting vs Actual)
    best_preds = test_preds["Gradient Boosting"]
    residuals = y_test - best_preds

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Actual vs Predicted
    axes[0].scatter(y_test, best_preds, alpha=0.5, color='#2b5c8f', edgecolors='none', s=30)
    max_val = max(y_test.max(), best_preds.max())
    min_val = min(y_test.min(), best_preds.min())
    axes[0].plot([min_val, max_val], [min_val, max_val], color='#e74c3c', linestyle='--', linewidth=2, label='Perfect Fit (y = x)')
    axes[0].set_title('Actual vs. Predicted Sale Price (Gradient Boosting)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Actual Sale Price ($)', fontsize=11)
    axes[0].set_ylabel('Predicted Sale Price ($)', fontsize=11)
    axes[0].legend()

    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color='#34495e', bins=35)
    axes[1].axvline(0, color='#e74c3c', linestyle='--', linewidth=1.5)
    axes[1].set_title('Residual Error Distribution (Actual - Predicted)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Residual Error ($)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)

    plt.tight_layout()
    chart3_path = os.path.join(charts_dir, '08_residuals_analysis.png')
    fig.savefig(chart3_path, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Saved residual analysis chart to: {chart3_path}")

    return results_df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    parser = argparse.ArgumentParser(description="Train and Evaluate Housing Valuation Models")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(project_root, "data", "processed", "Ames_Housing_Cleaned.csv"),
        help="Path to cleaned CSV dataset"
    )
    parser.add_argument(
        "--charts_dir",
        type=str,
        default=os.path.join(project_root, "outputs", "charts"),
        help="Directory to save evaluation charts"
    )
    args = parser.parse_args()
    train_and_evaluate_models(args.data_path, args.charts_dir)
