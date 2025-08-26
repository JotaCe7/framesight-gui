import io
import os
import requests

from PIL import Image
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="FrameSight UI",
    page_icon="🤖",
    layout="wide"
)

st.title("🖼️ FrameSight Object Detection")
st.write("Upload an imageto detect objects using the YOLOv8 model.")

# --- API Configuration ---
# Get the API URL from an environment variable, with a default for local development.
api_base_url = os.getenv("API_URL", "http://127.0.0.1:8000")
api_url = f"{api_base_url}/detect/"

# ---File Uploader ---
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Upload Image.", use_column_width=True)
    st.write("")

    # --- Inference Button ---
    if st.button("Detect Objects"):
        with st.spinner("Running inference..."):
            try:
                # Get the bytes from the upload file to send to the API.
                image_bytes = uploaded_file.getvalue()

                # The file needs to be sent as multipart/form-data.
                files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}

                # --- API Call ---
                response = requests.post(api_url, files=files)

                if response.status_code == 200:
                    # Get the processing time from the custom header
                    process_time = response.headers.get("X-Process-Time")
                    results = response.json()

                    if process_time:
                        st.success(f"Inference complete in {float(process_time):.2f} seconds.")
                    else:
                        st.success(f"Inference complete.")

                    # Display the rae JSON response for now
                    st.subheader("Detection Results (JSON)")
                    st.json(results)
                else:
                    st.error(f"Error from API: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API: {e}")
            except Exception as e:
                st.error(f"An unexpected error ocurred: {e}")