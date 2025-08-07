from src.preprocess import train_test_data
from sklearn.metrics import accuracy_score

def train_model(model, X, y):
    X_train, X_test, y_train, y_test = train_test_data(X, y)
    model.fit(X_train, y_train)

    # Predict the labels for the training data
    y_pred_train = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    # # Predict the labels for the test data
    # y_pred = model.predict(X_test)
    # accuracy_test = accuracy_score(y_test, y_pred)

    return model, accuracy_train


