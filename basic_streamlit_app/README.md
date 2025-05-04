# Automobile Database Explorer App

## Project Overview

This project is a basic Streamlit web application designed to allow users to explore and filter a dataset of automobile details. The app provides an interactive interface to browse and narrow down vehicle data based on specific criteria such as MPG, Number of Cylinders, and Horsepower.

The primary goal is to demonstrate a simple data exploration tool built with Streamlit, showcasing basic data loading, filtering based on user input from sliders and dropdowns, and dynamic display of results.

## Instructions

To run the Streamlit application (`automobile_app.py` - assuming this is the filename, please correct if different), follow these steps:

1.  **Clone the Repository**

2.  **Install Dependencies:** Ensure you have the necessary Python libraries installed. Use the `requirements.txt` file to ensure that the correct versions are installed.

3.  **Ensure the Dataset is Accessible:** Make sure the `fed_rd_year&gdp.csv` (or the correct filename for your automobile data) file is located in the **same directory** as your `automobile_app.py` script.

4.  **Run the Streamlit App:** Navigate into the directory containing the Python script and run the app:
    ```bash
    streamlit run automobile_app.py
    ```

    OR

    **Run in Streamlit Cloud:** [Link to Deployed App](https://petko-data-science-portfolio-74dezpd6ygwclv3ny2ifsk.streamlit.app/)

## Dataset Description

* **Source:** The dataset, "Automobile Database", is sourced from [Kaggle](https://www.kaggle.com/datasets/akshaydattatraykhare/car-details-dataset).
* **Content:** It contains details about various automobiles, including attributes like MPG, Number of Cylinders, and Horsepower, which are used for filtering within the app.

## App Functionality and Data Filtering

This application provides an interactive interface for exploring the automobile dataset:

1.  **Data Loading:** The application loads the automobile dataset from a CSV file.
2.  **Unfiltered Data Toggle:** Users can toggle a switch to show or hide the complete, unfiltered dataset dataframe for reference.
3.  **Filter Tabs:** The core filtering functionality is organized into tabs for different criteria:
    * **MPG Tab:** Allows users to select a minimum and maximum range for Miles Per Gallon using a slider.
    * **Number of Cylinders Tab:** Allows users to select a specific number of cylinders from a dropdown menu.
    * **Horsepower Tab:** Allows users to select a minimum and maximum range for Horsepower using a slider.
4.  **Dynamic Display:** Based on the user's selections in the filter tabs, the application dynamically displays a filtered version of the dataset dataframe.

## Outputs

The main output of this application is the interactive display of the automobile dataset, which is filtered in real-time based on the user's selections.

* **Filtered Dataframe:** A table showing the subset of vehicles that meet the criteria selected in the filter tabs.
* **Unfiltered Dataframe (Optional):** The complete dataset can be displayed via a toggle for comparison.



**Project Author:** Zach Petko *(Assuming Zach Petko is the author based on previous context)*
