from os import environ
from pathlib import Path

import __main__
import numpy as np
import pandas as pd
import streamlit as st
from dill import load
import scipy

# Fix pandas not found https://stackoverflow.com/a/65318623
__main__.pandas = pd
__main__.scipy = scipy

MODE = environ.get("MODE", "dev")

modelPath = Path("./server/Fifa_Model.pkl")
ciPath = Path("./server/ci.pkl")


def loadPkl(path: Path):
    obj = load(open(path, mode="rb"))
    return obj


model = loadPkl(modelPath)
ci = loadPkl(ciPath)
statParams = {"min_value": 0, "max_value": 100, "value": 0}
currParams = {"min_value": 0, "value": 0}

c = st.container()
st.markdown(
    """
    <style>
    .main {
        font-family: 'Orbitron', sans-serif;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        width: 100%;
        padding: 10px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

c.title("Fifa Overall Predictor")


# Input fields
moveReact = c.number_input("Movement Reactions", **statParams)
mentat = c.number_input("Mentality Composure", **statParams)
wage = c.number_input("Wage (EUR)", **currParams)
release = c.number_input("Release Clause (EUR)", **currParams)

col1, col2 = c.columns(2)
with col1:
    value = st.number_input("Value (EUR)", **currParams)
with col2:
    age = st.number_input("Age", min_value=18, max_value=120)

col3, col4, col5 = c.columns(3)
with col3:
    phys = st.number_input("Physic", **statParams)
with col4:
    pace = st.number_input("Pace", **statParams)
with col5:
    shoot = st.number_input("Shooting", **statParams)

col6, col7, col8 = c.columns(3)
with col6:
    pas = st.number_input("Passing", **statParams)
with col7:
    drib = st.number_input("Dribbling", **statParams)
with col8:
    defe = st.number_input("Defending", **statParams)


def predict():
    data = {
        "movement_reactions": moveReact,
        "mentality_composure": mentat,
        "wage_eur": wage,
        "release_clause_eur": release,
        "value_eur": value,
        "age": age,
        "physic": phys,
        "pace": pace,
        "shooting": shoot,
        "passing": pas,
        "dribbling": drib,
        "defending": defe,
    }
    if not all(
        [
            moveReact,
            mentat,
            wage,
            release,
            value,
            age,
            phys,
            pace,
            shoot,
            pas,
            drib,
            defe,
        ]
    ):
        st.error(
            "No inputs should be 0",
        )
        return

    data = pd.DataFrame(data, index=[0])
    prediction = model.predict(data)
    predictionFmted = int(np.floor(prediction[0]))
    ciStr = ci.ci(predictionFmted)
    c.markdown(
        f"""<h1 style='text-align: center'>{predictionFmted}</h1>
        <h4 style='text-align: center'>{ciStr}</h4>
        """,
        unsafe_allow_html=True,
    )


c.button("Predict", on_click=predict)
