from src.preprocess import train_test_data
from src.train import train_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def evaluate_model(model, X, y, accuracy_train, X_test, y_test):
    # model, accuracy_train = train_model(model, X, y)
    # X_train, X_test, y_train, y_test = train_test_data(X, y)
    
    # Predict the labels for the test data
    y_pred =model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred)

    # Compute Precision, recall, F1-Score and ROC AUC
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Create a metrics DataFrame
    df_metrics = pd.DataFrame({
        'Training Accuracy': [accuracy_train],
        'Test Accuracy': [accuracy_test],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1],
        'ROC AUC': [roc_auc]
    })

    # convert values to percentages
    for col in ['Training Accuracy',
                'Test Accuracy',
                'Precision',
                'Recall',
                'F1-Score',
                'ROC AUC']:
        df_metrics[col] = df_metrics[col].map(lambda x: f'{x:.2%}')
        
    # Print the DataFrame
    print(df_metrics)

    return df_metrics, y_pred

