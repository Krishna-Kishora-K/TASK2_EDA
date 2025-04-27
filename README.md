# Exploratory Data Analysis (EDA) - Titanic Dataset

## Objective:
The objective of this task is to perform **Exploratory Data Analysis (EDA)** on the Titanic Dataset using Python libraries such as Pandas, Matplotlib, Seaborn, and Plotly.

---

## Tools and Libraries Used:
- Python
- Pandas
- Matplotlib
- Seaborn
- Plotly

---

## Dataset:
- Titanic Dataset from Kaggle

---

## Steps Performed:

1. **Loading the Dataset**
   - Loaded the Titanic CSV file using `pandas.read_csv()`.

2. **Basic Data Exploration**
   - Displayed the first few rows using `.head()`.
   - Used `.info()` to understand data types and missing values.
   - Generated summary statistics using `.describe()`.

3. **Handling Missing Values**
   - Identified missing values using `.isnull().sum()`.

4. **Summary Statistics**
   - Calculated Mean, Median, Standard Deviation for numeric columns.

5. **Data Visualization**
   - Created histograms to understand feature distributions.
   - Created boxplots to detect outliers.
   - Generated correlation heatmaps.
   - Created pairplots to observe feature relationships.

6. **Pattern Identification**
   - Plotted survival rates across gender and passenger classes.
   - Observed skewness of numerical features.

---

## Important Visualizations:

- Histogram for Age and Fare
- Boxplot for Age and Fare
- Correlation Heatmap
- Pairplot colored by survival status
- Survival rate vs Gender and Pclass

---

## Observations:

- Females had a higher survival rate compared to males.
- First-class passengers had better survival rates.
- The 'Age' feature was slightly right-skewed.
- 'Fare' feature contained outliers (very high values).
- Some features like 'Age' and 'Cabin' had missing values.

---

## Conclusion:

This EDA process helped to understand:
- Data structure
- Important features
- Relationships between variables
- Presence of outliers and missing data

This will be helpful for future predictive modeling tasks.

---


