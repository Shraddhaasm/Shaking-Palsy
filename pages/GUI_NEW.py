# import streamlit as st
# import pydicom
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from io import BytesIO

# # Configure the page layout
# st.set_page_config(page_title="Parkinson Prediction System", layout="wide")

# # Custom CSS for Styling
# st.markdown("""
#     <style>
#         body {
#             background-color: #121212;
#             color: white;
#             font-family: 'Times New Roman', serif;
#         }
#         .main {
#             background-color: #121212;
#         }
#         .big-title {
#             font-size: 96px;
#             font-weight: bold;
#             text-align: center;
#             color: #ffcc00;
#             text-shadow: 4px 4px 7px rgba(255,255,255,0.4);
#             font-family: 'Times New Roman', serif;
#         }
#         .sub-header {
#             font-size: 30px;
#             font-weight: bold;
#             color: #00ffaa;
#             font-family: 'Times New Roman', serif;
#         }
#         .regular-text {
#             font-size: 22px;
#             font-family: 'Times New Roman', serif;
#         }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown(
#     """
#     <style>
#         [data-testid="stSidebar"] {display: none;} /* Hide sidebar */
#         [data-testid="stSidebarNavToggle"] {
#             visibility: hidden;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# if st.button("🏠 Home"):
#     st.switch_page("Landing_page.py")  # Ensure the filename matches your landing page

# # Page Title
# st.markdown('<p class="big-title">🧠 Parkinson Prediction System</p>', unsafe_allow_html=True)

# # Load the saved model with custom objects
# @st.cache_resource
# def load_model():
#     model_path = "C:/Users/Shraddha/PD PROJECT/efficientnetb0_model.keras"  # Update your path
#     try:
#         with tf.keras.utils.custom_object_scope({'weighted_sparse_categorical_crossentropy': weighted_sparse_categorical_crossentropy}):
#             model = tf.keras.models.load_model(model_path)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# # Custom loss function (same as training)
# def weighted_sparse_categorical_crossentropy(y_true, y_pred):
#     class_weights_tensor = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
#     y_true = tf.cast(y_true, tf.int32)
#     sample_weights = tf.gather(class_weights_tensor, y_true)
#     loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
#     return loss * sample_weights

# # Load the model
# model = load_model()

# # File uploader for DICOM files
# uploaded_file = st.file_uploader("📂 Upload a DICOM File (.dcm)", type=["dcm"])

# # Function to preprocess DICOM images
# def preprocess_dicom(dicom_data):
#     image = dicom_data.pixel_array
#     window_center = 50
#     window_width = 350
#     min_val = window_center - window_width / 2
#     max_val = window_center + window_width / 2

#     # Apply windowing
#     image = np.clip(image, min_val, max_val)
#     image = (image - min_val) / (max_val - min_val)
#     image = np.uint8(image * 255)

#     # Ensure proper shape
#     if len(image.shape) == 2:
#         image = np.expand_dims(image, axis=-1)

#     # Resize & convert to 3-channel
#     image = tf.image.resize(image, (224, 224))
#     image = tf.image.grayscale_to_rgb(image)
#     image = tf.keras.applications.resnet50.preprocess_input(image)

#     return np.expand_dims(image, axis=0)

# if uploaded_file is not None:
#     try:
#         # Read the uploaded DICOM file
#         dicom_data = pydicom.dcmread(BytesIO(uploaded_file.getvalue()))

#         # Extract image data
#         image_data = dicom_data.pixel_array
#         height, width = image_data.shape
#         center_x, center_y = width // 2, height // 2
#         radius = min(width, height) // 8  # Small Circle

#         # Layout: Image on Left, Metadata on Right
#         col1, col2 = st.columns([1, 1])

#         with col1:
#             st.markdown('<p class="sub-header">🖼️ DICOM Image</p>', unsafe_allow_html=True)
#             fig, ax = plt.subplots(figsize=(2, 2))
#             ax.imshow(image_data, cmap="gray")
#             circle = plt.Circle((center_x, center_y), radius, color="red", linewidth=1, fill=False)
#             ax.add_patch(circle)
#             ax.axis("off")
#             st.pyplot(fig)

#         with col2:
#             st.markdown('<p class="sub-header">📋 DICOM Metadata</p>', unsafe_allow_html=True)
#             metadata_keys = ["PatientName", "PatientID", "StudyDate", "Modality"]
#             for key in metadata_keys:
#                 if hasattr(dicom_data, key):
#                     st.markdown(f'<p class="regular-text"><b>{key}:</b> {getattr(dicom_data, key)}</p>', unsafe_allow_html=True)

#         # Button to Predict Parkinson's Disease
#         if st.button("🧠 Predict Parkinson's Disease"):
#             if model:
#                 # Preprocess and Predict
#                 image_input = preprocess_dicom(dicom_data)
#                 predictions = model.predict(image_input)
#                 class_idx = np.argmax(predictions, axis=1)[0]
#                 label_dict_inv = {0: '✅ CONTROL', 1: '⚠️ Parkinson\'s Detected', 2: '⚠️ Prodromal Stage'}
#                 prediction_result = label_dict_inv[class_idx]

#                 # Show prediction
#                 if class_idx == 0:
#                     st.success(prediction_result)
#                 else:
#                     st.error(prediction_result)
#             else:
#                 st.error("❌ Model not loaded properly. Please check the model path and retry.")
#     except Exception as e:
#         st.error(f"❌ Error reading DICOM file: {e}")



import streamlit as st
import pydicom
import numpy as np
import tensorflow as tf
#import cv2
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="Parkinson Prediction System", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
            font-family: 'Times New Roman', serif;
        }
        .main {
            background-color: #121212;
        }
        .big-title {
            font-size: 80px;
            font-weight: bold;
            text-align: center;
            color: #ffcc00;
            text-shadow: 4px 4px 7px rgba(255,255,255,0.4);
        }
        .sub-header {
            font-size: 30px;
            font-weight: bold;
            color: #00ffaa;
        }
        .regular-text {
            font-size: 22px;
        }
    </style>
""", unsafe_allow_html=True)

# Hide sidebar
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNavToggle"] {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

if st.button("🏠 Home"):
     st.switch_page("Landing_page.py")

st.markdown('<p class="big-title">🧠 Parkinson Prediction System</p>', unsafe_allow_html=True)

# -------------------------------
# Load the Pretrained Model
# -------------------------------
@st.cache_resource
def load_model():
    model_path = "C:/Users/Shraddha/PD PROJECT/efficientnetb0_model.keras"
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# -------------------------------
# DICOM Image Preprocessing
# -------------------------------
def preprocess_dicom(dicom_data):
    """Preprocess the uploaded DICOM image (without cv2)."""
    try:
        # Convert DICOM to float32 numpy array
        image = dicom_data.pixel_array.astype(np.float32)

        # Normalize to 0–255
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0
        image = image.astype(np.uint8)

        # Resize to (224, 224)
        image = tf.image.resize(image[..., np.newaxis], (224, 224)).numpy()

        # Convert grayscale to RGB by duplicating channels
        image = np.repeat(image, 3, axis=-1)

        # Preprocess for EfficientNet
        image = tf.keras.applications.efficientnet.preprocess_input(image)

        return np.expand_dims(image, axis=0)

    except Exception as e:
        st.error(f"❌ Error processing DICOM file: {e}")
        return None


# -------------------------------
# File Uploader for DICOM Files
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload a DICOM File (.dcm)", type=["dcm"])

if uploaded_file is not None:
    try:
        dicom_data = pydicom.dcmread(BytesIO(uploaded_file.getvalue()))
        image_data = dicom_data.pixel_array
        height, width = image_data.shape
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 8  # Small Circle

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown('<p class="sub-header">🖼️ DICOM Image</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(image_data, cmap="gray")
            circle = plt.Circle((center_x, center_y), radius, color="red", linewidth=1, fill=False)
            ax.add_patch(circle)
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.markdown('<p class="sub-header">📋 DICOM Metadata</p>', unsafe_allow_html=True)
            metadata_keys = ["PatientName", "PatientID", "StudyDate", "Modality"]
            for key in metadata_keys:
                if hasattr(dicom_data, key):
                    st.markdown(f'<p class="regular-text"><b>{key}:</b> {getattr(dicom_data, key)}</p>', unsafe_allow_html=True)

        # -------------------------------
        # Predict Button
        # -------------------------------
        if st.button("🧠 Predict Parkinson's Disease"):
            if model:
                image_input = preprocess_dicom(dicom_data)
                if image_input is not None:
                    predictions = model.predict(image_input)
                    class_idx = np.argmax(predictions, axis=1)[0]
                    label_dict_inv = {0: '✅ CONTROL', 1: '⚠️ Parkinson\'s Detected', 2: '⚠️ Prodromal Stage'}
                    prediction_result = label_dict_inv[class_idx]

                    if class_idx == 0:
                        st.success(prediction_result)
                    else:
                        st.error(prediction_result)
            else:
                st.error("❌ Model not loaded properly. Please check the model path and retry.")

    except Exception as e:
        st.error(f"❌ Error reading DICOM file: {e}")
