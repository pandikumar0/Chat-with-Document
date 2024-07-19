import streamlit as st
from PIL import Image
import os

from transformers import pipeline

# Initialize the pipeline
pipe = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

st.set_page_config(page_title="Chat with Document")       
st.header("Chat with Document")

# Initialize session state for the uploaded image path
if 'image_path' not in st.session_state:
    st.session_state['image_path'] = None

# Uploading the image
uploaded_file = st.file_uploader("Choose the Image", accept_multiple_files=False, type=['png', 'jpg', 'jpeg'])

# Display the uploaded image and save it locally
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)
    image_path = os.path.join("temp", uploaded_file.name)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save(image_path)
    st.session_state['image_path'] = image_path

# Input for the question
query = st.text_input("Question: ", key="input")

# Button to submit the question
submit = st.button("Ask Question")

# If an image is uploaded, a question is asked, and the button is pressed
if st.session_state['image_path'] and submit and query:
    # Call the pipeline with the local image path and query
    result = pipe(image=st.session_state['image_path'], question=query)
    
    # Display the answer
    st.write(result[0]['answer'])
