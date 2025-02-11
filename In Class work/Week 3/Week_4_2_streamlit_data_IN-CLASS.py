import streamlit as st
import pandas as pd

# To get browser in vscode
##  Go to search bar
##  show and run commands
##  simple browser show
##  put in local streamlit url


# ================================
# Step 1: Displaying a Simple DataFrame in Streamlit
# ================================

st.subheader("Now, let's look at some data!")

# Creating a simple DataFrame manually
# This helps students understand how to display tabular data in Streamlit.
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
})


t1,t2=st.tabs(['Manual Data', 'Imported Data'])

# Displaying the table in Streamlit
# st.dataframe() makes it interactive (sortable, scrollable)
t1.write("Here's a simple table:")
t1.dataframe(df)

# ================================
# Step 2: Adding User Interaction with Widgets
# ================================

# Using a selectbox to allow users to filter data by city
# Students learn how to use widgets in Streamlit for interactivity
city=t1.selectbox("Select a city", df['City'].unique())

# Filtering the DataFrame based on user selection
filtered_df=df[df['City']==city]
# Display the filtered results
t1.markdown(f"### People in {city}.")
t1.dataframe(filtered_df)

# ================================
# Step 3: Importing Data Using a Relative Path
# ================================

# Now, instead of creating a DataFrame manually, we load a CSV file
# This teaches students how to work with external data in Streamlit
# # Ensure the "data" folder exists with the CSV file
# Display the imported dataset
df2=pd.read_csv('.\Data\sample_data.csv')
t2.dataframe(df2)
# Using a selectbox to allow users to filter data by city
salary=city2=t2.slider("Select a min salary.", min_value=df2['Salary'].min(), max_value=df2['Salary'].max())
# Students learn how to use widgets in Streamlit for interactivity

# Filtering the DataFrame based on user selection
filtered_df2=df2[df2['Salary']>=salary]
# Display the filtered results
t2.markdown(f'### Salaries over {salary}.')

t2.dataframe(filtered_df2)
# ================================
# Summary of Learning Progression:
# 1️⃣ Displaying a basic DataFrame in Streamlit.
# 2️⃣ Adding user interaction with selectbox widgets.
# 3️⃣ Importing real-world datasets using a relative path.
# ================================