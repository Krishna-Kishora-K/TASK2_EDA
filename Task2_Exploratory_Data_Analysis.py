import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px

# Load the CSV file
df = pd.read_csv('Titanic_dataset_new.csv')

# Show first 5 rows
df.head()

# See data types and non-null counts
df.info()

# See basic statistical summary
df.describe()

# See missing values
df.isnull().sum()

# Mean
print("Mean values:\n", df.mean(numeric_only=True))

# Median
print("\nMedian values:\n", df.median(numeric_only=True))

# Standard Deviation
print("\nStandard Deviation:\n", df.std(numeric_only=True))

df.hist(figsize=(12, 8), bins=20)
plt.tight_layout()
plt.show()

# Correlation
corr = df.corr(numeric_only=True)

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

sns.pairplot(df, hue="Survived")
plt.show()

# Skewness
print(df.skew(numeric_only=True))

# Survived vs Gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Gender')
plt.show()

# Survival based on Passenger Class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()

# Age Distribution of Survivors vs Non-Survivors
sns.histplot(data=df, x='Age', hue='Survived', kde=True)
plt.title('Age Distribution by Survival')
plt.show()
