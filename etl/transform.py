
# This file cleans and preprocesses raw data for ML model

import pandas as pd  # data manipulation library
from sklearn.preprocessing import LabelEncoder  # encode categorical columns to numbers

def transform_data(df):
    # drop duplicate rows if any
    df = df.drop_duplicates()

    # drop rows where any value is null
    df = df.dropna()

    # convert one-hot month columns to a single Month integer column if needed
    month_columns = [col for col in df.columns if col.startswith("Month_") and col != "Month"]
    if "Month" not in df.columns and month_columns:
        month_order = {
            "Month_Jan": 1,
            "Month_Feb": 2,
            "Month_Mar": 3,
            "Month_Apr": 4,
            "Month_May": 5,
            "Month_Jun": 6,
            "Month_Jul": 7,
            "Month_Aug": 8,
            "Month_Sep": 9,
            "Month_Oct": 10,
            "Month_Nov": 11,
            "Month_Dec": 12,
        }
        df['Month'] = df[month_columns].idxmax(axis=1).map(lambda x: month_order.get(x, 0))
        no_month = df[month_columns].sum(axis=1) == 0
        if no_month.any():
            df.loc[no_month, 'Month'] = 0
        df = df.drop(columns=month_columns)

    # convert one-hot visitor type columns to a single VisitorType column if needed
    visitor_columns = [col for col in df.columns if col.startswith("VisitorType_") and col != "VisitorType"]
    if "VisitorType" not in df.columns and visitor_columns:
        if "VisitorType_Returning_Visitor" in df.columns:
            df['VisitorType'] = df['VisitorType_Returning_Visitor'].astype(int)
        elif "VisitorType_New_Visitor" in df.columns:
            df['VisitorType'] = (1 - df['VisitorType_New_Visitor'].astype(int)).astype(int)
        else:
            df['VisitorType'] = 0
        df = df.drop(columns=visitor_columns)

    # if Month is categorical text, encode it to integers
    if "Month" in df.columns and df['Month'].dtype == object:
        le_month = LabelEncoder()
        df['Month'] = le_month.fit_transform(df['Month'])

    # if VisitorType is categorical text, encode it to integers
    if "VisitorType" in df.columns and df['VisitorType'].dtype == object:
        le_visitor = LabelEncoder()
        df['VisitorType'] = le_visitor.fit_transform(df['VisitorType'])

    # convert Weekend column to integer (True → 1, False → 0)
    if 'Weekend' in df.columns:
        df['Weekend'] = df['Weekend'].astype(int)

    # convert Revenue column to integer if present
    if 'Revenue' in df.columns:
        df['Revenue'] = df['Revenue'].astype(int)

    print(f"Transformed data shape: {df.shape}")  # log final shape
    return df  # return cleaned dataframe

if __name__ == "__main__":
    from extract import extract_data  # import extract function
    df = extract_data()  # fetch raw data
    df = transform_data(df)  # apply transformations
    print(df.head())  # print first 5 rows to verify
    print(df.dtypes)  # print column data types to verify encoding