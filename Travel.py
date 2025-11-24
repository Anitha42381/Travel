import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="TC & PRS Prediction App", layout="wide")

st.title("TC & PRS Prediction App")
st.write("Upload your file and run prediction")

# ===============================
# 1️⃣ FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # ===============================
    # 2️⃣ USER INPUT SELECTION
    # ===============================
    st.subheader("Select Columns for Analysis")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.write("**Numeric Columns:** ", num_cols)
    st.write("**Categorical Columns:** ", cat_cols)

    x_col = st.selectbox("Choose feature column", df.columns)
    y_col = st.selectbox("Choose target column", df.columns)

    # ===============================
    # 3️⃣ ANALYSIS OUTPUT
    # ===============================
    st.subheader("Basic Statistics")

    if x_col in num_cols:
        st.write("**Numeric Summary:**")
        st.write(df[x_col].describe())
    else:
        st.write("**Value Counts:**")
        st.write(df[x_col].value_counts())

    # ===============================
    # 4️⃣ SIMPLE VISUALIZATION
    # ===============================
    import matplotlib.pyplot as plt

    st.subheader("Visualization")

    fig, ax = plt.subplots()
    
    if x_col in num_cols:
        ax.hist(df[x_col].dropna())
        ax.set_title(f"Distribution of {x_col}")
    else:
        df[x_col].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Counts of {x_col}")

    st.pyplot(fig)

    # ===============================
    # 5️⃣ MODEL PREDICTION (OPTIONAL)
    # ===============================
    model_path = "model.pkl"

    if os.path.exists(model_path):
        st.subheader("Prediction")
        model = pickle.load(open(model_path, "rb"))

        if st.button("Run Prediction"):
            try:
                preds = model.predict(df)
                df["Prediction"] = preds
                st.write(df.head())
                st.success("Prediction completed!")
            except Exception as e:
                st.error(f"Model error: {e}")
    else:
        st.info("No model.pkl found — upload one to enable predictions.")
