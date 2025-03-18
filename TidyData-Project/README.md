# Tidy Data Project: Federal R&D Spending 

## Project Overview
This project aims to clean and make tidy data gathered on Federal R&D Spending. After making the data tidy, visualizations and a pivot table is created to provide a preliminary analysis of Federal R&D Spending. Tidy data fits these criteria:
- Each variable forms a column.
- Each observation forms a row.
- Each type of observational unit forms a table.

This project contributes to my data science portfolio by showcasing my ability to clean, tidy, and visualize real-world data, and enhancing my data analysis skills.

## Instructions

To run the Jupyter Notebook (`Tidy_Data_Project.ipynb`) and reproduce the analysis, follow these steps:

1.  **Clone the Repository**
2.  **Install Dependencies**
    Ensure you have pandas and matplotlib Python libraries installed.
3.  **Run the Jupyter Notebook**
4.  **Execute Cells:**

## Dataset Description

-   **Source:** The dataset "Federal R&D Budgets" (`fed_rd_year&gdp.csv`) was adapted from a GitHub repository.
-   **Content:** It contains federal Research and Development (R&D) spending as a percentage of GDP, broken down by department and year.
-   **Preprocessing:** The initial data was in a wide format, with each year's spending as a separate column. The preprocessing involved renaming columns, melting the data to a long format, and converting the year column to numeric.

## Data Cleaning and Tidy Process

The data was transformed to meet tidy data principles:

1.  **Loading Data:** The raw CSV file was loaded into a pandas DataFrame.
2.  **Renaming Columns:** Column names were simplified to just the year.
3.  **Melting Data:** The DataFrame was melted to create `department`, `Year`, and `Spending` columns.
4.  **Numeric Year:** The `Year` column was converted to a numeric data type.

## Visualizations

### 1. Time Series Line Chart

This chart displays the spending trends over time for each department, allowing for easy comparison of spending patterns.

![Time Series Line Chart](path/to/your/time_series_chart.png)  *(Replace with actual image path or remove)*

```python
for dept in fed_data_clean['department'].unique():
    dept_data = fed_data_clean[fed_data_clean['department'] == dept]
    plt.plot(dept_data['Year'], dept_data['Spending'], label=dept)

plt.title('Spending by Department')
plt.xlabel('Year')
plt.ylabel('Spending')
plt.legend(fontsize='small')
plt.show()
