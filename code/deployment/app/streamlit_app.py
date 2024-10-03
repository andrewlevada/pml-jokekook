import streamlit as st
import requests

# Input
input_text = st.text_input("A joke about...")

# Button
if st.button("Generate a joke"):
    prompt = f"A joke about {input_text}: "
    response = requests.post("http://api:8000/generate", json={"prompt": prompt})
    st.write(f"{response.json()['generated_text']}")
