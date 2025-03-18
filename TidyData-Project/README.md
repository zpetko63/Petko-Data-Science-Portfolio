
# Tidy Data Project: Federal R&D Spending 

## Project Overview
This project aims to clean and make tidy data gathered on Federal R&D Spending. After making the data tidy, visualizations and a pivot table is created to provide a preliminary analysis of Federal R&D Spending. [Tidy data](https://vita.had.co.nz/papers/tidy-data.pdf) fits these criteria:
- Each variable forms a column.
- Each observation forms a row.
- Each type of observational unit forms a table.

This project contributes to my data science portfolio by showcasing my ability to clean, tidy, and visualize real-world data, and enhancing my data analysis skills.

## Instructions

To run the Jupyter Notebook (`Tidy_Data_Project.ipynb`) and reproduce the analysis, follow these steps:

1.  **Clone the Repository**
2.  **Install Dependencies:** Ensure you have pandas and matplotlib Python libraries installed.
3.  **Run the Jupyter Notebook**

## Dataset Description

-   **Source:** The dataset "Federal R&D Budgets" (`fed_rd_year&gdp.csv`) was adapted from a [GitHub repository](https://github.com/rfordatascience/tidytuesday/tree/main/data/2019/2019-02-12).
-   **Content:** It contains federal Research and Development (R&D) spending, broken down by department and year in wide format.

## Data Cleaning and Tidying Process

The data was transformed to be tidy using the following steps:

1.  **Load Data:** The raw CSV file was loaded into a pandas DataFrame.
2.  **Rename Columns:** Column names were simplified to just the year instead of the given long name.
3.  **Melting Data:** The DataFrame was melted to create `department`, `Year`, and `Spending` columns.
4.  **Numeric Year:** The `Year` column was converted to a numeric data type.

## Visualizations

### 1. Time Series Spending Line Chart

This chart illustrates the trends in federal R&D spending over time for each department from the dataset. Each department is represented by a distinct line and color, allowing for easy comparison across departments.

![Time Series Spending Line Chart](Spending_Line.png)  

### 2. Spending Share Bar Chart

This bar chart provides a snapshot of the proportion of federal R&D spending by department for a specific year (2017 in this analysis). The chart represents each department’s share as a percentage of total spending, helping users quickly grasp how funding is allocated across departments.

![Spending Share Bar Chart](Spending_Share_Bar.png)

## Pivot Table

The pivot table summarizes the median department spending for each year in the dataset, providing a robust measure of central tendency in spending data. Unlike averages, the median is less sensitive to extreme values, offering a clearer picture of typical spending patterns.




