"""Streamlit front end for the SuperKart sales forecast model.

Loads the model committed to the repository (model/best_model.joblib), collects the
inputs, stores them in a DataFrame and shows the predicted product-store sales total.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model" / "best_model.joblib"
REFERENCE_YEAR = 2025

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="centered")


@st.cache_resource
def load_model():
    """Load the trained scikit-learn pipeline once per session."""
    return joblib.load(MODEL_PATH)


st.title("SuperKart Sales Forecast")
st.write("Enter product and store details to predict the total sales revenue "
         "for that product in that store.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run the training pipeline first.")
    st.stop()

model = load_model()

PRODUCT_TYPES = ["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
                 "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
                 "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"]

with st.form("inputs"):
    st.subheader("Product details")
    c1, c2 = st.columns(2)
    product_weight = c1.number_input("Product weight", 1.0, 30.0, 12.5, 0.1)
    allocated_area = c2.number_input("Allocated display area (ratio)", 0.0, 0.5, 0.07, 0.001, format="%.3f")
    product_mrp = c1.number_input("Product MRP", 20.0, 300.0, 150.0, 0.5)
    sugar = c2.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
    product_type = c1.selectbox("Product type", PRODUCT_TYPES)
    category = c2.selectbox("Product category (Id prefix)", ["FD", "DR", "NC"],
                            help="FD = food, DR = drinks, NC = non-consumable")

    st.subheader("Store details")
    c3, c4 = st.columns(2)
    store_year = c3.number_input("Store establishment year", 1980, REFERENCE_YEAR, 2000, 1)
    store_size = c4.selectbox("Store size", ["Small", "Medium", "High"])
    city = c3.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = c4.selectbox("Store type", ["Departmental Store", "Food Mart",
                                             "Supermarket Type1", "Supermarket Type2"])
    submitted = st.form_submit_button("Predict sales")

if submitted:
    # Inputs are saved into a DataFrame with the same columns the model was trained on
    input_df = pd.DataFrame([{
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": allocated_area,
        "Product_Type": product_type,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": city,
        "Store_Type": store_type,
        "Product_Category": category,
        "Store_Age": REFERENCE_YEAR - int(store_year),
    }])
    st.write("Input data sent to the model:")
    st.dataframe(input_df, hide_index=True)
    prediction = float(model.predict(input_df)[0])
    st.success(f"Predicted product-store sales total: **{prediction:,.2f}**")
