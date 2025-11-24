import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import json

st.set_page_config(page_title="Travel Package Recommendation", layout="wide")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("finaltraveldata.csv")
    return df

df = load_data()

# ----------------------------
# Preprocessing Setup
# ----------------------------
cat_cols = ['From_City', 'Destination', 'Destination_Type', 
            'Budget_Range', 'Accommodation_Type', 'Transport_Mode', 
            'Meal_Plan', 'Activity_Types', 'Season', 
            'Package_Type', 'Recommended_For']

num_cols = ['Trip_Duration_Days', 'Approx_Cost', 'Activity_Count']

# Fit encoders
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(df[cat_cols])

scaler = MinMaxScaler()
scaler.fit(df[num_cols])

# Combine features
cat_features = ohe.transform(df[cat_cols])
num_features = scaler.transform(df[num_cols])
cdata = np.hstack([num_features, cat_features])

# Fit Nearest Neighbors model
model = NearestNeighbors(n_neighbors=5, metric="cosine")
model.fit(cdata)

# ----------------------------
# Title
# ----------------------------
st.title("🌍 Travel Package Recommendation System")
st.write("Provide your travel preferences to get personalized package recommendations.")

# ----------------------------
# User Input Form
# ----------------------------
with st.form("user_input_form"):

    col1, col2 = st.columns(2)

    with col1:
        From_City = st.selectbox("From City", df["From_City"].unique())
        Destination = st.selectbox("Destination", df["Destination"].unique())
        Destination_Type = st.selectbox("Destination Type", df["Destination_Type"].unique())
        Budget_Range = st.selectbox("Budget Range", df["Budget_Range"].unique())
        Accommodation_Type = st.selectbox("Accommodation Type", df["Accommodation_Type"].unique())
        Transport_Mode = st.selectbox("Transport Mode", df["Transport_Mode"].unique())

    with col2:
        Meal_Plan = st.selectbox("Meal Plan", df["Meal_Plan"].unique())
        Season = st.selectbox("Season", df["Season"].unique())
        Activity_Types = st.selectbox("Activity Types", df["Activity_Types"].unique())

        Trip_Duration_Days = st.slider(
            "Trip Duration (Days)", 
            int(df["Trip_Duration_Days"].min()), 
            int(df["Trip_Duration_Days"].max()),
            5
        )

        Activity_Count = st.slider(
            "Activity Count", 
            int(df["Activity_Count"].min()),
            int(df["Activity_Count"].max()),
            3
        )

        Approx_Cost = st.number_input(
            "Approx Cost (₹)", 
            min_value=float(df["Approx_Cost"].min()),
            max_value=float(df["Approx_Cost"].max()),
            value=20000.0
        )

    submitted = st.form_submit_button("🔍 Get Recommendations")

# ----------------------------
# Recommendation Logic
# ----------------------------
if submitted:
    user_row = pd.DataFrame([[
        From_City, Destination, Destination_Type, Trip_Duration_Days,
        Budget_Range, Approx_Cost, Accommodation_Type, Transport_Mode,
        Meal_Plan, Activity_Count, Activity_Types, Season,
        "Standard", "Family"  # Dummy fields for compatibility
    ]], columns=df.columns)

    # Encode
    user_cat = ohe.transform(user_row[cat_cols])
    user_num = scaler.transform(user_row[num_cols])
    user_vector = np.hstack([user_num, user_cat])

    # Find neighbors
    distances, indices = model.kneighbors(user_vector)

    # Build results DataFrame
    recommendations = df.iloc[indices[0]].copy()
    recommendations["Similarity"] = (1 - distances[0]).round(3)

    st.subheader("🎯 Top Recommended Packages")
    st.dataframe(recommendations)

    # ----------------------------
    # JSON Output
    # ----------------------------
    json_output = recommendations.to_dict(orient="records")
    st.download_button(
        label="📥 Download Recommendations (JSON)",
        data=json.dumps(json_output, indent=4),
        file_name="recommendations.json",
        mime="application/json"
    )

    # ----------------------------
    # Image Display (optional)
    # ----------------------------
    try:
        img_data = pd.read_csv("travel_packages_with_images.csv")

        st.subheader("📸 Destination Images")

        for idx in indices[0][:3]:
            row = img_data.loc[idx]
            st.image(row["Destination_Image_URL"], width=350, caption=row["Destination"])
    except:
        st.info("No image file found. Upload travel_packages_with_images.csv for image support.")
