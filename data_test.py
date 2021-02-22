import pandas as pd
import numpy as np
white_wine_file = "winequality-white.csv"

 # semicolon delimited 
df = pd.read_csv(white_wine_file, header = 0)
# put the original column names in a python list
features = list(df.columns[:11])
target = df['quality']
X = df[features]
y = target
print (X)
print (y)