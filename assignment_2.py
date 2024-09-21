import streamlit as st
import pandas as pd

# Load your dataset
file_path = 'healthcare_dataset.csv'  # Update with the path to your dataset
data = pd.read_csv(file_path)

# Title of the application
st.title("Cognitive HealthCare Application")

# Introduction section
st.write("""
## Welcome to the Cognitive HealthCare Application
This application allows you to explore and analyze healthcare data. Below, you can interact with patient data and filter based on various criteria.
""")

# Show data overview
st.write("### Patient Data Overview")
st.dataframe(data)

# Adding filters for interactive exploration
st.sidebar.header("Filter Options")

# Filter by Age
age = st.sidebar.slider("Select Age Range", int(data['Age'].min()), int(data['Age'].max()), (20, 80))
filtered_data = data[(data['Age'] >= age[0]) & (data['Age'] <= age[1])]

# Filter by Medical Condition
condition = st.sidebar.multiselect(
    "Select Medical Condition",
    options=data['Medical Condition'].unique(),
    default=data['Medical Condition'].unique()
)
filtered_data = filtered_data[filtered_data['Medical Condition'].isin(condition)]

# Filter by Gender
gender = st.sidebar.multiselect(
    "Select Gender",
    options=data['Gender'].unique(),
    default=data['Gender'].unique()
)
filtered_data = filtered_data[filtered_data['Gender'].isin(gender)]

# Display the filtered data
st.write("### Filtered Patient Data")
st.dataframe(filtered_data)

# Display some basic statistics
st.write("### Basic Statistics")
st.write(filtered_data.describe())

# Visualization section
st.write("### Visualizations")

# Bar chart for medical conditions
st.bar_chart(filtered_data['Medical Condition'].value_counts())

# Histogram for Age
st.write("#### Age Distribution")
st.bar_chart(filtered_data['Age'])

# Pie chart for Gender distribution
gender_counts = filtered_data['Gender'].value_counts()
st.write("#### Gender Distribution")
st.write(gender_counts)
st.pyplot(gender_counts.plot.pie(autopct="%1.1f%%").figure)
