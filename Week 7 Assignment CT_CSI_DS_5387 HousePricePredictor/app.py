# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Load California housing data and select relevant columns
def load_real_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df = df[['AveRooms', 'MedHouseVal']]  # Using average rooms to predict house value
    df.rename(columns={'AveRooms': 'average_rooms', 'MedHouseVal': 'median_value'}, inplace=True)
    return df

# Train linear regression model on the real dataset
def train_model():
    df = load_real_data()
    X = df[['average_rooms']]
    y = df['median_value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, df

# Streamlit app interface
def main():
    st.title('🏡 California Housing Price Estimator')
    st.write('Enter the average number of rooms to estimate the median house value.')

    model, df = train_model()

    rooms = st.number_input('Average number of rooms', min_value=1.0, max_value=15.0, value=5.0, step=0.1)

    if st.button('Predict House Value'):
        prediction = model.predict([[rooms]])
        st.success(f'Estimated Median House Value: ${prediction[0]*100000:.2f}')  # scale for display

        # Plot data and prediction
        fig = px.scatter(df, x='average_rooms', y='median_value', title='Rooms vs Median House Value')
        fig.add_scatter(x=[rooms], y=[prediction[0]], mode='markers',
                        marker=dict(size=12, color='red'), name='Your Input')
        st.plotly_chart(fig)

# Run app
if __name__ == '__main__':
    main()
