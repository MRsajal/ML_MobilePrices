import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
import numpy as np
import re
import matplotlib.pyplot as plt

def predict_mobile_price():
    df_model=pd.read_csv('Data.csv')


    # Features: RAM, Storage, Original_Price
    # features = ['RAM', 'Storage', 'Original_Price']
    features = ['RAM', 'Storage', 'Original_Price', 'Brand', 'RAM_Storage_Ratio', 'Log_Original_Price']
    target = 'Used_Price'
    
    
    # Filter out any rows that still have NaN in features after cleaning
    df_model.dropna(subset=features, inplace=True)
    print('Number of records after cleaning and dropping missing values:', len(df_model))
    #print(df_model.head())

    X = df_model[features]
    y = df_model[target]

    if len(X) == 0:
        print("Error: After cleaning and dropping rows with missing values, no data remains for training.")
        return

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Model Selection and Training (Random Forest Regressor)
    print("Training Random Forest Regressor...")
    #model = RandomForestRegressor(n_estimators=956, random_state=42,min_samples_leaf=1,max_depth=19, n_jobs=-1)
    model = RandomForestRegressor(n_estimators=253, random_state=42,min_samples_leaf=3,max_depth=20, n_jobs=-1)
    # model=XGBRegressor(
    #     n_estimators=600,
    #     learning_rate=0.05,
    #     max_depth=10,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     random_state=42
    # )
    model.fit(X_train, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\nModel Evaluation:")
    # The '\$' is correct; it prints a literal dollar sign before the number.
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    print("\nThe model has been trained and evaluated.")


        
    # param_grid={
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [5, 10, None],
    #     'min_samples_leaf': [3, 5, 10]
    # }

    # grid_search=GridSearchCV(
    #     estimator=RandomForestRegressor(random_state=42,n_jobs=-1),
    #     param_grid=param_grid,
    #     scoring='neg_mean_squared_error',
    #     cv=5,
    #     verbose=1,
    #     n_jobs=-1
    # )
    # grid_search.fit(X_train,y_train)
    # print("\nBest Hyperparameters from Grid Search:")
    # print(grid_search.best_params_)


    # param_dist = {
    #     'n_estimators': randint(100, 1000),
    #     'max_depth': randint(5, 25),
    #     'min_samples_leaf': randint(1, 10)
    # }

    # search = RandomizedSearchCV(
    #     RandomForestRegressor(random_state=42, n_jobs=-1),
    #     param_distributions=param_dist,
    #     n_iter=30,
    #     cv=3,
    #     scoring='r2',
    #     random_state=42
    # )
    # search.fit(X_train, y_train)
    # print("Best params:", search.best_params_)


    # param_dist={
    #     'n_estimators': randint(50,300),
    #     'max_depth': [5,10,15,20,None],
    #     'min_samples_leaf': randint(3,20)
    # }

    # random_search=RandomizedSearchCV(
    #     estimator=RandomForestRegressor(random_state=42,n_jobs=-1),
    #     param_distributions=param_dist,
    #     n_iter=20,
    #     scoring='neg_mean_squared_error',
    #     cv=5,
    #     verbose=1,
    #     n_jobs=-1,
    #     random_state=42
    # )
    # random_search.fit(X_train,y_train)
    # print("\nBest Hyperparameters from Randomized Search:")
    # print(random_search.best_params_)

    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='dodgerblue', edgecolor='k')
    plt.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--', linewidth=2, label='Perfect Prediction')

    # Add labels and title
    plt.xlabel("Actual Prices", fontsize=12)
    plt.ylabel("Predicted Prices", fontsize=12)
    plt.title("Actual vs Predicted Used Mobile Prices", fontsize=14)
    plt.legend()

    # Display evaluation metrics on the plot
    textstr = '\n'.join((
        f'MAE  : {mae:.2f}',
        f'RMSE : {rmse:.2f}',
        f'R²    : {r2:.2f}'
    ))
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    
predict_mobile_price()