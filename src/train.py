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

from preprocess import preprocess_data, train_test_data
from dataload import read_data
from sklearn.metrics import accuracy_score

df = read_data()     

independent_vars, dependent_var = preprocess_data(df)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_data(independent_vars, dependent_var)

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
    # Evaluate on the training set using cross-validation
    df_metrics = classification_cv(model, X=X_train, y=y_train)
    results[name] = df_metrics



best_model, scores_df  = select_best_model(results)
model = models[best_model] # actucal model object 
 
 
def train_best_model(X=independent_vars, y=dependent_var,  best_model=model):
    # Fit best model on training data
    trained = best_model.fit(X_train, y_train)
    # Evaluate on hold-out test set
    y_pred = trained.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    # Save the trained model
    joblib.dump(trained, Model_loc / f"{best_model.__class__.__name__}.joblib")
    print(f"Best model {best_model} trained and saved successfully. Test accuracy: {test_acc:.4f}")
 
train_best_model()
