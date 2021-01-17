import pandas as pd
df = pd.read_pickle("currentCompiled_0.csv")
print(len(df))
print(df.head())