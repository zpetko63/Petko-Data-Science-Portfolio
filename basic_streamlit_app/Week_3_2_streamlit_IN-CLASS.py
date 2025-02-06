# Import the Streamlit library

import streamlit as st

# Navigate using terminal
## ls - shows all files in current directory
## cd (next folder) - changes current directory to specified folder (cd .. goes back a folder)

# Activate streamlit
## streamlit run (python file name)

# Display a simple text message
st.title("Hello, Streamlit!")
st.write("This is my first streamlit app!")
st.markdown("## This is my first streamlit app!")
# Display a large title on the app

# ------------------------
# INTERACTIVE BUTTON
# ------------------------

# Create a button that users can click.
# If the button is clicked, the message changes.

if st.button("Click me!"):
    st.write("You clicked the button!")
else:
    st.write("Click the button!")


slider_value = st.slider('Slide me', min_value=0, max_value=10)
st.write(slider_value)






#create tabs
tab1, tab2 = st.tabs(["Button", "Slider"])

# create button counter on tab1
tab1.markdown('## Click the button!')

if 'button_count' not in st.session_state:
    st.session_state.button_count = 0

if tab1.button("Click"):
  st.session_state.button_count+=1
  tab1.write(f"You clicked the button {st.session_state.button_count} times!")

if tab1.button("Reset Counter"):
    st.session_state.button_count=0

# create slider on tab2
slider_value = tab2.slider('Slide', min_value=0, max_value=10)
tab2.write(f"The slider is at {slider_value}")

# ------------------------
# COLOR PICKER WIDGET
# ------------------------

# Creates an interactive color picker where users can choose a color.
# The selected color is stored in the variable 'color'.

# Display the chosen color value

# ------------------------
# ADDING DATA TO STREAMLIT
# ------------------------

# Import pandas for handling tabular data

# Display a section title

# Create a simple Pandas DataFrame with sample data


# Display a descriptive message

# Display the dataframe in an interactive table.
# Users can scroll and sort the data within the table.

# ------------------------
# INTERACTIVE DATA FILTERING
# ------------------------

# Create a dropdown (selectbox) for filtering the DataFrame by city.
# The user selects a city from the unique values in the "City" column.

# Create a filtered DataFrame that only includes rows matching the selected city.

# Display the filtered results with an appropriate heading.
  # Show the filtered table

# ------------------------
# NEXT STEPS & CHALLENGE
# ------------------------

# Play around with more Streamlit widgets or elements by checking the documentation:
# https://docs.streamlit.io/develop/api-reference
# Use the cheat sheet for quick reference:
# https://cheat-sheet.streamlit.app/

### Challenge:
# 1️⃣ Modify the dataframe (add new columns or different data).
# 2️⃣ Add an input box for users to type names and filter results.
# 3️⃣ Make a simple chart using st.bar_chart().