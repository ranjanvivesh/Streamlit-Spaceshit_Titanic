#LOAD THE SAVED MODEL 
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from features import engineer

st.set_page_config(page_title="Spaceship Titanic", page_icon="🚀")

@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / "model.joblib")

model = load_model()

#TITLE AND SIDE BAR INPUT 
st.title("🚀 Spaceship Titanic")
st.write("Estimate whether a passenger was transported to another dimension.")

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

st.sidebar.header("Passenger details")
home_planet = st.sidebar.selectbox("Home planet", ["Earth", "Europa", "Mars"])
destination = st.sidebar.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
deck = st.sidebar.selectbox("Cabin deck", list("ABCDEFGT"))
side = st.sidebar.selectbox("Cabin side", ["P", "S"])
age = st.sidebar.slider("Age", 0, 80, 28)
cryo = st.sidebar.checkbox("In cryosleep")
vip = st.sidebar.checkbox("VIP")

spend = {
    col: 0.0 if cryo else st.sidebar.number_input(col, min_value=0.0, value=0.0, step=10.0)
    for col in SPEND_COLS
}

#ROW AND PREDICTION
row = pd.DataFrame([{
    "HomePlanet": home_planet,
    "CryoSleep": cryo,
    "Cabin": f"{deck}/0/{side}",
    "Destination": destination,
    "Age": age,
    "VIP": vip,
    **spend,
}])

if st.button("Predict"):
    proba = model.predict_proba(engineer(row))[0, 1]
    st.metric("Probability transported", f"{proba:.0%}")
    st.progress(float(proba))
    if proba >= 0.5:
        st.success("Likely transported")
    else:
        st.error("Likely not transported")