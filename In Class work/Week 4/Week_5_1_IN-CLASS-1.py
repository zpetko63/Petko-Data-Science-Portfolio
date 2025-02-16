import pandas as pd            # Library for data manipulation
import seaborn as sns          # Library for statistical plotting
import matplotlib.pyplot as plt  # For creating custom plots
import streamlit as st         # Framework for building interactive web apps

# ================================================================================
#Missing Data & Data Quality Checks
#
# This lecture covers:
# - Data Validation: Checking data types, missing values, and ensuring consistency.
# - Missing Data Handling: Options to drop or impute missing data.
# - Visualization: Using heatmaps and histograms to explore data distribution.
# ================================================================================
st.title("Missing Data & Data Quality Checks")
st.markdown("""
This lecture covers:
- **Data Validation:** Checking data types, missing values, and basic consistency.
- **Missing Data Handling:** Options to drop or impute missing data.
- **Visualization:** Using heatmaps and histograms to understand data distribution.
""")

# ------------------------------------------------------------------------------
# Load the Dataset
# ------------------------------------------------------------------------------
# Read the Titanic dataset from a CSV file.
df = pd.read_csv("titanic.csv")

# ------------------------------------------------------------------------------
# Display Summary Statistics
# ------------------------------------------------------------------------------
# Show key statistical measures like mean, standard deviation, etc.
st.write("**Summary Statistics**")
st.write(df.shape)
st.dataframe(df.describe())

# ------------------------------------------------------------------------------
# Check for Missing Values
# ------------------------------------------------------------------------------
# Display the count of missing values for each column.
st.write("**Number of Missing Values by Column**")
st.dataframe(df.isnull().sum())

# ------------------------------------------------------------------------------
# Visualize Missing Data
# ------------------------------------------------------------------------------
# Create a heatmap to visually indicate where missing values occur.
st.write("Heatmap of missing data")
fig, ax = plt.subplots()
sns.heatmap(df.isnull(),cmap='viridis', cbar=False)
st.pyplot(fig)

# ================================================================================
# Interactive Missing Data Handling
#
# Users can select a numeric column and choose a method to address missing values.
# Options include:
# - Keeping the data unchanged
# - Dropping rows with missing values
# - Dropping columns if more than 50% of the values are missing
# - Imputing missing values with mean, median, or zero
# ================================================================================
st.subheader("Handle Missing Data")

# Work on a copy of the DataFrame so the original data remains unchanged.
column = st.selectbox('Choose a column to fill', df.select_dtypes(include=["number"]).columns)
# Apply the selected method to handle missing data.
st.dataframe(df[column])


method = st.radio("Select a Method", 
         ["Original DF", "Drop Rows", "Drop Columns (>50% Missing)", 
          "Impute Mean", "Impute Median", "Impute Zero"])

# Copy our original dataframe
df_clean=df.copy()# you have to use this method to not affect the origional



if method == "Original DF":
    pass 
elif method == "Drop Rows":
    df_clean = df_clean.dropna()
elif method == "Drop Columns (>50% Missing)":
    df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isnull().mean()>0.5])
elif method == "Impute Mean":
    df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
elif method == "Impute Median":
    df_clean[column] = df_clean[column].fillna(df_clean[column].median())
elif method == "Impute Zero":
    df_clean[column] = df_clean[column].fillna(0)

st.dataframe(df_clean)


# ------------------------------------------------------------------------------
# Compare Data Distributions: Original vs. Cleaned
#
# Display side-by-side histograms and statistical summaries for the selected column.
# ------------------------------------------------------------------------------
