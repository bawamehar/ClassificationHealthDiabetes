from dataload import read_data
from sklearn.model_selection import train_test_split

print("preprocess.py loaded")

def preprocess_data(df):
    df['Diabetes_012'] =df['Diabetes_012'].replace({2:1})
    df = df.rename(columns = {'Diabetes_012': 'Diabetes_binary'})
    target = 'Diabetes_binary'
    independent_vars = df.drop(target, axis=1)
    dependent_var = df[target]
    df.head(5)
    
    return independent_vars, dependent_var

# def train_test_data(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=25)
#     return X_train, X_test, y_train, y_test

