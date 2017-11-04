# This is where we manipulate the data so that it can be consumed be the model
import pandas as pd

# Read the txt doc
df = pd.read_csv('iris.txt', sep=',', header=None)
columns = ['s_length', 's_width', 'p_length', 'p_width', 'class']
df.columns = columns

# Estimate the highest and lowest bounds
dataHigh = 10.
dataLow = 0.001

# What we want to normalise it to
normaliseHigh = 1
normaliseLow = 0

# Get the ranges
dataRange = dataHigh - dataLow
normalisedRange = normaliseHigh - normaliseLow


# Normalise the value
def normalise_val(value):
    norm = (value - dataLow) * normalisedRange
    norm = norm / dataRange
    return norm + normaliseLow


# Convert name to value
def normalise_iris(val):
    if 'setosa' in val:
        return [1, 0, 0]
    if 'versicolor' in val:
        return [0, 1, 0]
    if 'virginica' in val:
        return [0, 0, 1]


# Normalise the entire table
for col in columns:
    if col != 'class':
        # We normalise the values into another table
        df[col] = df[col].apply(normalise_val)

df['setosa'] = df['class'].map({'Iris-setosa': 1, 'Iris-versicolor': 0, 'Iris-virginica': 0})
df['versicolor'] = df['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 0})
df['virginica'] = df['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 1})

# Drop the irrelevant column
df = df.drop('class', axis=1)

# Save the dataframe to a file
df.to_csv('norm_iris.csv', sep=',')
