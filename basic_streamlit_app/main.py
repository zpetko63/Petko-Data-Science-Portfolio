# Basic Streamlit App - Automobile Database
# Zach Petko
# 2-9-2025



#Import Libraries
import streamlit as st
import pandas as pd

#Run Streamlit
##streamlit hello


#Import and Format Dataset
auto = pd.read_csv('.\Data\Automobile_data.csv')
auto = auto.loc[:,['name', 'mpg', 'cylinders', 'horsepower']]

#App Structure
st.markdown('# Automobile Database\n##### This app allows the user to filter and view data from an automobile dataset sourced from kaggle (more information in README).')


# Show Datafram Toggle
df_toggle = st.checkbox('Show Unfiltered Dataframe')
if df_toggle:
    st.dataframe(auto)



# Create Tabs
st.markdown('### Select a tab below to filter on that variable.')
mpg_tab, cyl_tab, hp_tab = st.tabs(['MPG', 'Cylinders', 'Horsepower'])


# MPG Tab
mpg_slider_values = mpg_tab.slider('Select a MPG Range', min_value=auto['mpg'].min(), max_value=auto['mpg'].max(), value=(auto['mpg'].min(),auto['mpg'].max()))
mpg_auto = auto[(auto['mpg']>mpg_slider_values[0]) & (auto['mpg']<mpg_slider_values[1])].sort_values(by='mpg')
mpg_tab.dataframe(mpg_auto)


#Cylinders Tab
cyl_select=cyl_tab.selectbox('Select a number of cylinders',auto['cylinders'].unique())
cyl_auto = auto[auto['cylinders']==cyl_select].sort_values(by='cylinders')
cyl_tab.dataframe(cyl_auto)


# Horsepower Tab
hp_slider_values = hp_tab.slider('Select a Horsepower Range', min_value=auto['horsepower'].min(), max_value=auto['horsepower'].max(), value=(auto['horsepower'].min(),auto['horsepower'].max()))
hp_auto = auto[(auto['horsepower']>hp_slider_values[0]) & (auto['horsepower']<hp_slider_values[1])].sort_values(by='horsepower')
hp_tab.dataframe(hp_auto)

