

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Generates synthetic dataset for house sizes and corresponding prices
def generate_house_data(n_samples=100):
    np.random.seed(42)
    size = np.random.normal(1500, 500, n_samples)
    price = size * 100 + np.random.normal(0, 10000, n_samples)
    return pd.DataFrame({'size_sqft': size, 'price': price})

# Trains a simple linear regression model on the generated dataset
def train_model():
    df = generate_house_data()
    X = df[['size_sqft']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Streamlit application interface for user interaction and model inference
def main():
    st.title('🏠 Simple House Pricing Predictor')
    st.write('Input house size (in square feet) to estimate its market price.')

    # Train model on synthetic dataset
    model = train_model()

    # Accept user input for house size
    size = st.number_input('House size (square feet)', min_value=500, max_value=5000, value=1500)

    # Display prediction and visualization on button click
    if st.button('Predict price'):
        prediction = model.predict([[size]])
        st.success(f'Estimated price: ${prediction[0]:,.2f}')

        # Generate new data for plotting
        df = generate_house_data()
        fig = px.scatter(df, x='size_sqft', y='price', title='Size vs Price Relationship')
        fig.add_scatter(x=[size], y=[prediction[0]], mode='markers',
                        marker=dict(size=15, color='red'),
                        name='Prediction')
        st.plotly_chart(fig)

# Entry point for the Streamlit app
if __name__ == '__main__':
    main()
