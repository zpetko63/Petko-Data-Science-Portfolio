# This is a basic Streamlit app for an automobile database

## Data Source: Automobile Database - ([Kaggle](https://www.kaggle.com/datasets/akshaydattatraykhare/car-details-dataset))

## App Summary: 
This app lets users explore and filter the automobile database bsed on their desired criteria. The user has the ability to choose between MPG, Number of Cylinders, Horsepower, or just the unfiltered data set to narrow down results. 

## App Functionality:
- Unfiltered Data Toggle: Users can select the toggle titled "Show Unfiltered Dataframe" to choose between showing the dataframe above the filter tabs. This can be useful for comparing data.
- Filter Tabs: The app has 3 tabs that the user can select from to filter on its respective variable. The options are MPG, number of Cylinders, and Horsepower.
  - MPG Tab: This tab allows the user to select a minimum and maximum MPG value using a slider. The vehicles that satisfy the inputed criteria will appear in the displayed dataframe below.
  - Number of Cylinders Tab: This tab allows the user to select a single cylinder count using a single value dropdown. The vehicles that satisfy the inputed criteria will appear in the displayed dataframe below.
  - Horsepower Tab: This tab allows the user to select a minimum and maximum Horsepower value using a slider. The vehicles that satisfy the inputed criteria will appear in the displayed dataframe below.
