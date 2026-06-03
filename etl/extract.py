
# This file extracts raw data from MySQL database

import pandas as pd  # data manipulation library
from sqlalchemy import create_engine  # database connection engine
from dotenv import load_dotenv  # load environment variables from .env file
import os  # access environment variables

load_dotenv()  # load .env file variables into environment

def get_engine():
    # build mysql connection string using env variables
    url = (
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)  # return sqlalchemy engine object

def extract_data():
    csv_path = os.getenv('DATA_PATH')
    if csv_path:
        csv_path = os.path.expandvars(os.path.expanduser(csv_path))
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, low_memory=False)
            print(f"Loaded {len(df)} rows from CSV at {csv_path}")
            return df
        else:
            print(f"DATA_PATH is set but file not found: {csv_path}")

    engine = get_engine()  # get database connection
    df = pd.read_sql("SELECT * FROM online_shoppers", engine)  # fetch all rows from table
    print(f"Extracted {len(df)} rows")  # log how many rows fetched
    return df  # return dataframe

if __name__ == "__main__":
    df = extract_data()  # run extraction
    print(df.head())  # print first 5 rows to verify