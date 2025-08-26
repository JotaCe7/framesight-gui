import io
import os
import requests

from PIL import Image, ImageDraw
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="FrameSight UI",
    page_icon="🤖",
    layout="wide"
)

st.title("🖼️ FrameSight Object Detection")
st.write("Use your camera to detect objects in real-time with the YOLOv8 model.")

# --- API Configuration ---
# Get the API URL from an environment variable, with a default for local development.
api_base_url = os.getenv("API_URL", "http://127.0.0.1:8000")
api_url = f"{api_base_url}/detect/"

# ---Camera Input ---
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Read image file buffer as bytes:
    bytes_data = img_file_buffer.getvalue()

    # Convert to a PIL Image object:
    image = Image.open(io.BytesIO(bytes_data))

    st.image(image, caption="Your camera Feed.", use_column_width=True)
    st.write("")
    with st.spinner("Running inference..."):
        try:
            # The file needs to be sent as multipart/form-data.
            files = {"file": ("frame.jpg", bytes_data, "image/jpeg")}

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

                # --- Draw Bounding Boxes ----
                detections = results.get("detections", [])
                if detections:
                    draw = ImageDraw.Draw(image)
                    for detection in detections:
                        confidence = detection['confidence']

                        # Check confidence threshold
                        if confidence > 0.75: # harcoded for now
                            box = detection['box']
                            label = f"{detection['class_name']}  ({confidence:.2f})"

                            # Draw the bounding box
                            draw.rectangle(box, outline="red", width=3)

                            # Draw the label
                            text_position = (box[0], box[1] - 10) # TODO: set font size
                            draw.text(text_position, label, fill="red")

                    st.subheader("Detected Objects")
                    st.image(image, use_column_width=True)

                # Keep displaying the rae JSON response for debugging
                st.subheader("Detection Results (JSON)")
                st.json(results)
            else:
                st.error(f"Error from API: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the API: {e}")
        except Exception as e:
            st.error(f"An unexpected error ocurred: {e}")

