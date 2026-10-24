import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error

def load_and_prepare_data(gen_path, prices_path):
    print("Loading Parquet files...")
    generation = pd.read_parquet(gen_path)
    prices = pd.read_parquet(prices_path)

    print("Merging and cleaning data...")
    df = pd.merge(generation, prices, on=['Start date', 'End date'], how='inner')

    price_col = 'Germany/Luxembourg [€/MWh] Calculated resolutions'
    features = [
        'Wind onshore [MWh] Calculated resolutions',
        'Wind offshore [MWh] Calculated resolutions',
        'Photovoltaics [MWh] Calculated resolutions',
        'Fossil gas [MWh] Calculated resolutions'
    ]

    cols_to_clean = features + [price_col]
    for col in cols_to_clean:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')

    df['Price_Spike'] = (df[price_col] > 100).astype(int)
    
    # Keep the exact price for regression
    model_data = df[features + [price_col, 'Price_Spike']].dropna()
    
    X = model_data[features]
    y_clf = model_data['Price_Spike']
    y_reg = model_data[price_col]
    
    return X, y_clf, y_reg, features

def train_and_evaluate(X, y_clf, y_reg):
    print("Splitting data and training models...")
    X_train, X_test, y_train_c, y_test_c, y_train_r, y_test_r = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42
    )

    # Train Classifier (Spike Prediction)
    print("\n--- Training Classifier ---")
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train, y_train_c)
    clf_preds = clf_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test_c, clf_preds))

    # Train Regressor (Exact Price Prediction)
    print("\n--- Training Regressor ---")
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_train_r)
    reg_preds = reg_model.predict(X_test)
    
    r2 = r2_score(y_test_r, reg_preds)
    mae = mean_absolute_error(y_test_r, reg_preds)
    print(f"Regression R2 Score: {r2:.4f}")
    print(f"Regression Mean Absolute Error: {mae:.2f} €/MWh")
    
    return clf_model, reg_model

if __name__ == "__main__":
    gen_path = 'cleaned_generation.parquet'
    prices_path = 'cleaned_prices.parquet'
    
    X, y_clf, y_reg, feature_names = load_and_prepare_data(gen_path, prices_path)
    best_clf, best_reg = train_and_evaluate(X, y_clf, y_reg)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_clf, os.path.join('models', 'price_spike_model.pkl'))
    joblib.dump(best_reg, os.path.join('models', 'price_value_model.pkl'))
    joblib.dump(feature_names, os.path.join('models', 'model_features.pkl'))
    
    print("\nPipeline complete. Classifier and Regressor models exported.")