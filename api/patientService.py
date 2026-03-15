import pandas as pd
import psycopg2

DB_params = {
    "dbname" : "postgres",
    "user" : "postgres",
    "password" : "password",
    "host" : "localhost",
    "port" : "5432"
}

TABLE_NAME = "diabetic_patients"

def insert_patient(stlmdata):
    try:
        conn = psycopg2.connect(**DB_params)
        cursor = conn.cursor()
        
        create_query = f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                bmi FLOAT,
                age FLOAT,
                genhlth FLOAT,
                income FLOAT,
                highbp FLOAT,
                highchol FLOAT,
                education FLOAT
            );
            """
        cursor.execute(create_query)
        
        insert_query = f"""
        INSERT INTO {TABLE_NAME} (bmi, age, genhlth, income, highbp, highchol, education)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        cursor.execute(insert_query, stlmdata)
        conn.commit()
        print("Data successfully ingested ")

    except Exception as e:
            print("Error:", e)

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()