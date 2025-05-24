import os
from dotenv import load_dotenv
import streamlit as st

st.title("Hello")
# st.write("Werite text")
# st.text("text ")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
st.write(f"Your API key is: {api_key}")


ans=st.selectbox("Slect ",['First','Second','Third'])
st.write(f"Your selected value is {ans}")

st.warning("Your Scess Message")
if st.button("Submit"):
    st.success("Submitted")

if st.checkbox("Checkbox"):
    st.success("Checkbox clicked")

type_radio=st.radio('check any radio',['First','Second','Third'])
st.write(f"Your selected radio is {type_radio}")

slider=st.slider('slider',0,5)
st.write(f"Your slider is {slider}")