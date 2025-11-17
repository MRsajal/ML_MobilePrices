import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import numpy as np
import re
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import matplotlib.pyplot as plt
from scipy.stats import randint
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

def clean_numeric_column(series):
    def convert_to_gb(value):
        if pd.isna(value):
            return np.nan
        value = str(value).upper().strip().replace(',', '')
        if 'GB' in value:
            return float(re.sub(r'[^0-9.]', '', value.replace('GB', '')))
        elif 'MB' in value:
            # Simple conversion, assumes value is the number of MB
            return float(re.sub(r'[^0-9.]', '', value.replace('MB', ''))) / 1024
        else:
             # Try direct conversion if no units found
            try:
                return float(re.sub(r'[^0-9.]', '', value))
            except ValueError:
                return np.nan

    return series.apply(convert_to_gb)


def create_features(df):
    """Create additional features that might improve prediction accuracy"""
    df = df.copy()
    
    # Feature 1: Depreciation rate (how much value is lost)
    df['Depreciation_Rate'] = (df['Original_Price'] - df['Used_Price']) / df['Original_Price']
    
    # Feature 2: Price per GB of RAM
    df['Price_per_RAM_GB'] = df['Original_Price'] / df['RAM']
    
    # Feature 3: Price per GB of Storage
    df['Price_per_Storage_GB'] = df['Original_Price'] / df['Storage']
    
    # Feature 4: RAM to Storage ratio
    df['RAM_Storage_Ratio'] = df['RAM'] / df['Storage']
    
    # Feature 5: Total memory score (weighted combination)
    df['Memory_Score'] = df['RAM'] * 2 + df['Storage'] * 0.1
    
    # Feature 6: High-end phone indicator (based on original price)
    df['Is_High_End'] = (df['Original_Price'] > df['Original_Price'].quantile(0.75)).astype(int)
    
    # Feature 7: Log transformations for skewed data
    df['Log_Original_Price'] = np.log1p(df['Original_Price'])
    df['Log_RAM'] = np.log1p(df['RAM'])
    df['Log_Storage'] = np.log1p(df['Storage'])
    
    return df


def remove_outliers(X, y, method='iqr', factor=1.5):
    """Remove outliers using IQR method"""
    if method == 'iqr':
        # Calculate IQR for target variable
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Filter outliers
        mask = (y >= lower_bound) & (y <= upper_bound)
        return X[mask], y[mask]
    
    return X, y


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def predict_mobile_price():
    # Load and combine data
    df1 = pd.read_csv('mobile_prices_filled.csv')
    df2 = pd.read_csv('mobile_prices_filled2.csv')
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Standardize column names
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]
    
    # Clean numeric columns
    for col in ['RAM', 'Storage', 'Used_Price', 'Original_Price']:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    # Remove rows with missing target or original price
    df.dropna(subset=['Used_Price'], inplace=True)
    df_model = df.dropna(subset=['Original_Price']).copy()
    
    # Basic features cleaning
    features = ['RAM', 'Storage', 'Original_Price']
    df_model.dropna(subset=features, inplace=True)
    
    print(f"Data shape after initial cleaning: {df_model.shape}")
    
    # Create additional features
    df_model = create_features(df_model)
    
    # Enhanced feature set
    enhanced_features = [
        'RAM', 'Storage', 'Original_Price', 
        'Depreciation_Rate', 'Price_per_RAM_GB', 'Price_per_Storage_GB',
        'RAM_Storage_Ratio', 'Memory_Score', 'Is_High_End',
        'Log_Original_Price', 'Log_RAM', 'Log_Storage'
    ]
    
    # Remove any rows with NaN in new features
    df_model.dropna(subset=enhanced_features, inplace=True)
    print(f"Data shape after feature engineering: {df_model.shape}")
    
    X = df_model[enhanced_features]
    y = df_model['Used_Price']
    
    if len(X) == 0:
        print("Error: No data remains after cleaning.")
        return
    
    # Remove outliers
    X_clean, y_clean = remove_outliers(X, y)
    print(f"Data shape after outlier removal: {X_clean.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=None
    )
    
    # Model comparison
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=300, 
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42, 
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'Scaled Random Forest': Pipeline([
            ('scaler', RobustScaler()),
            ('rf', RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])
    }
    
    results = {}
    best_model = None
    best_score = float('-inf')
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse = np.sqrt(-cv_scores.mean())
        print(f"CV RMSE: {cv_rmse:,.2f} (+/- {np.sqrt(cv_scores.std() * 2):,.2f})")
        
        # Train and evaluate
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics
        
        if metrics['R2'] > best_score:
            best_score = metrics['R2']
            best_model = model
    
    # Feature importance for best model
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps['rf'], 'feature_importances_'):
        importances = best_model.named_steps['rf'].feature_importances_
    else:
        importances = None
    
    if importances is not None:
        feature_importance = pd.DataFrame({
            'feature': enhanced_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance (Top 5):")
        for _, row in feature_importance.head().iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY OF ALL MODELS:")
    print(f"{'='*50}")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  MAE: {metrics['MAE']:,.2f}")
        print(f"  RMSE: {metrics['RMSE']:,.2f}")
        print(f"  R²: {metrics['R2']:.4f}")
        print()
    
    return best_model, results


def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for Random Forest"""
    print("\nPerforming hyperparameter tuning...")
    
    param_dist = {
        'n_estimators': [200, 300, 400],
        'max_depth': [10, 15, 20, 25],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print("\nBest Hyperparameters:")
    for param, value in random_search.best_params_.items():
        print(f"{param}: {value}")
    
    return random_search.best_estimator_


if __name__ == "__main__":
    best_model, results = predict_mobile_price()
    
    # Uncomment the following lines if you want to perform hyperparameter tuning
    # print("\n" + "="*50)
    # print("HYPERPARAMETER TUNING")
    # print("="*50)
    # tuned_model = hyperparameter_tuning(X_train, y_train)
    # tuned_metrics = evaluate_model(tuned_model, X_test, y_test, "Tuned Random Forest")

