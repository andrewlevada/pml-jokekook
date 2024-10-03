import streamlit as st
import requests

# input
input_text = st.text_input("A joke about...")

if st.button("Generate a joke"):
    prompt = f"A joke about {input_text}: "
    # send prompt to api    
    response = requests.post("http://api:8000/generate", json={"prompt": prompt})
    # output
    st.write(f"{response.json()['generated_text']}")
