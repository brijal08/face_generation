import streamlit as st
import requests
import json
from io import BytesIO
from PIL import Image

st.title('Face image generation using cGAN')

# Text input for label
label = st.text_input("Enter Label:")

if st.button("Generate Image"):
    if label:
        # Send the label to the server
        response = requests.post(
            "https://brijal08-facegeneration.hf.space/generate",  # Replace with your API endpoint
            json={"label": label}
        )

        if response.status_code == 200:
            # Convert the response content to an image
            image = Image.open(BytesIO(response._content))
            # Display the image
            st.image(image, caption=label, use_column_width=True)
        else:
            st.write("Error: Unable to get the response from the server.")
    else:
        st.write("Please enter a label.")
