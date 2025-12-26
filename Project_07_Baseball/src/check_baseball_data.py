# check_baseball_data.py
# in-class data exploration

import pandas as pd

bb = pd.read_csv("baseball.csv")

print("=== BASIC INFO ===")
bb.info()

print("\n=== HEAD ===")
print(bb.head())

print("\n=== TAIL ===")
print(bb.tail())

print("\n=== Missing Values ===")
print(bb.isna().sum())

print("\n=== INDEX(ROWS) ===")
print(list(bb.index))

print("\n=== COLUMNS ===")
print(list(bb.columns))

print(list(bb.isna().sum()))
print(bb.isna().sum().sum())

print("\nUnique players:", bb['id'].nunique())
print("Unique teams:", bb['team'].nunique())
print("Unique leagues:", bb['lg'].dropna().unique())
