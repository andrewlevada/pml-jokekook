import streamlit as st

# Input
input_text = st.text_input("Enter key words")

# Button
if st.button("Generate a joke"):
    # Output
    output_text = input_text + " world"
    st.write(f"{output_text}")