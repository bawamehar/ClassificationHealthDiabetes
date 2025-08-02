import pandas as pd

print("dataload.py loaded")

def read_data():
    file_path = r"F:\Diabetes Project\ClassificationHealthDiabetes\diabetes_project\data\diabetes.csv"
    df = pd.read_csv(file_path)
    return df