import joblib
import numpy as np
import pandas as pd
import os
from dataload import read_data
from preprocess import preprocess_data
from evaluate import classification_cv, select_best_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_user_input(feature_names):
    """
    Prompt user for each feature value and return as a DataFrame.
    """
    input_data = {}
    for feature in feature_names:
        val = input(f"Enter value for {feature}: ")
        try:
            val = float(val)
        except ValueError:
            pass
        input_data[feature] = [val]
    return pd.DataFrame(input_data)


def main():
    # 1. Load and preprocess data
    df = read_data()
    X, y = preprocess_data(df)

    # 2. Train all models and evaluate using classification_cv
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=52),
        'Random Forest': RandomForestClassifier(random_state=52),
        'Gradient Boosting': GradientBoostingClassifier(random_state=52),
        'AdaBoost': AdaBoostClassifier(random_state=52),
        'XGBoost': XGBClassifier(eval_metric='logloss')
    }
    results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name}")
        df_metrics = classification_cv(model, X, y)
        results[name] = df_metrics

    print(df.head(5))
    # 3. Select best model using weighted metric
    best_model_name, scores_df = select_best_model(results)
    best_model = models[best_model_name]
    best_model.fit(X, y)
    # Save the best model
    model_path = os.path.join(os.path.dirname(__file__), '../models/AdaBoostClassifier.joblib')
    joblib.dump(best_model, model_path)
    print(f"Best model: {best_model_name} (saved to {model_path})")

    # 4. Get user input and predict
    feature_names = list(X.columns)
    
    print("Please enter the following features:")
    user_df = get_user_input(feature_names)
    prediction = best_model.predict(user_df)
    print(f"Predicted Diabetes (0=No, 1=Yes): {prediction[0]}")

if __name__ == "__main__":
    main()
