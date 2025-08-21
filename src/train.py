from evaluate import select_best_model
from pathlib import Path
import joblib
from evaluate import classification_cv

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
)
from xgboost import XGBClassifier

from preprocess import preprocess_data
from dataload import read_data

df = read_data()     

independent_vars, dependent_var = preprocess_data(df)

BASE_DIR = Path(__file__).resolve().parent.parent
Model_loc = BASE_DIR / "models"


models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree':DecisionTreeClassifier(random_state=52),
    'Random Forest':  RandomForestClassifier(random_state=52),
    'gradient boost': GradientBoostingClassifier(random_state=52),
    'adaboost':  AdaBoostClassifier(random_state=52),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}



results = {}

for name, model in models.items():
    print(f"\nEvaluating {name}")
    df_metrics = classification_cv(model)
    results[name] = df_metrics



best_model, scores_df  = select_best_model(results)
model = models[best_model] # actucal model object 


def train_best_model(X=independent_vars, y=dependent_var,  best_model=model):
    model = best_model.fit(X, y)
    #model_path = Model_loc/best_model.joblib"
    joblib.dump(model, Model_loc / f"{best_model.__class__.__name__}.joblib")
    #joblib.dump(model, Model_loc/best_model.joblib)
    print(f"Best model {best_model} trained and saved successfully.")

train_best_model()
