# Import necessary libraries
import numpy as np
import streamlit as st
import pandas as pd
from tensorflow.keras.preprocessing import image
import cv2
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import json
import time
import tensorflow as tf  # For TFLite model inference

# --- Path to your quantized TFLite model (stored in GitHub repo) ---
MODEL_PATH = "inceptionv3_quantized.tflite"  # <-- adjust path if different

# --- Check that the file exists ---
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

# --- Load the TFLite model ---
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Function to safely load Lottie animations from a URL ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            st.warning(f"⚠️ Could not load Lottie animation from: {url}")
            return None
        return r.json()
    except Exception as e:
        st.error(f"Error loading Lottie: {e}")
        return None

# --- Load local Lottie animation for trash concept ---
with open("src/Home_Robot.json", "r", encoding="utf-8") as f:
    trash_lottie = json.load(f)

# --- Custom CSS for neon-style navigation bar ---
st.markdown("""
    <style>
    .nav-pills {
        border: 2px solid #39ff14;
        border-radius: 12px;
        padding: 5px;
        background-color: black;
        box-shadow: 0 0 5px #39ff14, 0 0 10px #39ff14, 0 0 20px #39ff14;
        transition: 0.3s;
    }
    .nav-pills:hover {
        box-shadow: 0 0 10px #39ff14, 0 0 20px #39ff14, 0 0 30px #39ff14, 0 0 50px #39ff14;
    }
    .nav-pills .nav-link {
        color: #39ff14 !important;
        font-weight: bold;
    }
    .nav-pills .nav-link.active {
        background-color: #111 !important;
        border: 2px solid #39ff14;
        box-shadow: 0 0 10px #39ff14, 0 0 20px #39ff14, 0 0 30px #39ff14;
    }
    </style>
""", unsafe_allow_html=True)

# --- Adjust padding for Streamlit layout ---
st.markdown("""
    <style>
    div.block-container {padding-top: 5rem;}
    </style>
""", unsafe_allow_html=True)

# --- Main navigation menu ---
selected = option_menu(
    menu_title=None,
    options=["Home", "Data Analysis", "Image Classification"],
    icons=["house", "bar-chart", "gear"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# --- Header ---
st.title('Garbage Image Classifier')

# ------------------------- HOME PAGE -------------------------
if selected == "Home":
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.write("A garbage image classifier that can automatically identify and categorize waste into types such as plastic, metal, glass, paper, and organic...")
        st.markdown("### Business Use Cases:")
        st.markdown(""" 
        * Smart Recycling Bins  
        * Municipal Waste Management  
        * Educational Tools  
        * Environmental Analytics  
        """)
        st.markdown("### Dataset and Dataset Explanation:")
        st.markdown(""" 
        * Description: Dataset contains images of six categories — cardboard, glass, metal, paper, plastic, trash.  
        * Size: ~2,467 images  
        * Source: [Kaggle Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
        """)
    with right_col:
        st_lottie(trash_lottie, speed=1, loop=True, height=300)

# --------------------- DATA ANALYSIS PAGE ---------------------
elif selected == "Data Analysis":
    st.markdown("<style>.block-container {padding-bottom: 1rem; padding-left: 2rem; padding-right: 2rem;}</style>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 4.8], gap="large")
    with col1:
        with st.container():
            st.markdown('<div class="block-container">', unsafe_allow_html=True)
            selected = option_menu(
                menu_title="Main Menu",
                options=["Images Per Class", 
                         "Pixel intensity distribution for each class", 
                         "Example Images"],
                icons=["bar-chart", "graph-up", "image"],
                menu_icon="cast",
                default_index=0,
                orientation="vertical"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        if selected == "Images Per Class":
            img_col, text_col = st.columns([2, 2], gap='large')
            with img_col:
                categories = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
                counts = [403, 501, 410, 594, 482, 137]
                fig = px.bar(
                    x=categories, 
                    y=counts, 
                    title="📊 Number of Images Per Class",
                    text=counts,
                    color=categories,
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(xaxis_title="Class", yaxis_title="Number of Images", title_x=0.2, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            with text_col:
                st.markdown("### 📊 Distribution of Images Across Categories")
                st.markdown("- Paper has the highest number of images (~600)")
                st.markdown("- Glass and Plastic ~500 each")
                st.markdown("- Metal and Cardboard ~400 each")
                st.markdown("- Trash has the lowest (~130)")
                st.markdown("**Key Insight:** Dataset is imbalanced; consider augmentation or class weighting.")

        elif selected == "Pixel intensity distribution for each class":
            selected = option_menu(
                menu_title=None,
                options=["trash", "plastic", "glass", "metal", "paper", "cardboard"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal"
            )
            img_path = f"EDA Images\\Example Images\\{selected.capitalize()}.png"
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256]).flatten()
            hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256]).flatten()
            hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256]).flatten()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(256), y=hist_r, mode='lines', name='Red', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=np.arange(256), y=hist_g, mode='lines', name='Green', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=np.arange(256), y=hist_b, mode='lines', name='Blue', line=dict(color='blue')))
            fig.update_layout(title=f"Pixel intensity distribution for {selected.capitalize()} Images",
                              xaxis_title="Pixel value", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"Pixel distribution for '{selected}' across RGB channels.")

        elif selected == "Example Images":
            selected = option_menu(
                menu_title=None,
                options=["trash", "plastic", "glass", "metal", "paper", "cardboard"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal",
            )
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                image = Image.open(f"EDA Images\\Example Images\\{selected.capitalize()}.png")
                st.image(image, caption=f"Example image for {selected}", use_container_width=True)

# ------------------- IMAGE CLASSIFICATION PAGE -------------------
elif selected == "Image Classification":
    col1, col2 = st.columns([1.2, 4.8], gap="large")
    with col1:
        with st.container():
            st.markdown('<div class="block-container">', unsafe_allow_html=True)
            selected = option_menu(
                menu_title="Main Menu",
                options=["Image Classification", "Model performances"],
                icons=["bi-cpu", "graph-up"],
                menu_icon="cast",
                default_index=0,
                orientation="vertical"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        if selected == "Image Classification":
            dic = {0 : "cardboard", 3: "Paper", 1: "Glass", 2: "Metal", 4: "Plastic", 5: "Trash"}
            recycling_info = {
                "cardboard": "Can be recycled in most curbside programs. Remove any tape or non-paper materials.",
                "Glass": "Rinse and recycle in designated glass bins. Colored glass may have restrictions.",
                "Metal": "Clean and place in metal recycling bins. Includes aluminum cans and tin containers.",
                "Paper": "Recycle paper products. Avoid greasy or heavily soiled paper.",
                "Plastic": "Rinse and recycle plastics labeled 1-7 where accepted. Avoid bags in curbside bins.",
                "Trash": "Non-recyclable waste. Dispose of in regular trash bins."
            }

            with open("src/Loading.json", "r", encoding="utf-8") as f:
                loading_lottie = json.load(f)

            uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert("RGB")
                col1_disp, col2_disp = st.columns(2, gap="large")
                with col1_disp:
                    st.image(img, caption="Uploaded Image", use_container_width=True)
                with col2_disp:
                    placeholder = st.empty()
                    with placeholder.container():
                        st_lottie(loading_lottie, speed=1, loop=True, height=200, key="loading")
                        st.markdown("<p style='text-align:center;'>Analyzing image... please wait ⏳</p>", unsafe_allow_html=True)

                # --- Preprocess image for TFLite model ---
                img_resized = img.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
                img_array = np.array(img_resized, dtype=np.uint8)
                img_array = np.expand_dims(img_array, axis=0)

                time.sleep(1.5)

                # --- Run inference with TFLite ---
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])[0]

                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class]
                class_name = dic[predicted_class]

                with col2_disp:
                    placeholder.empty()
                    with placeholder.container():
                        st.subheader("Prediction Details")
                        st.write(f"**Class:** {class_name}")
                        st.write(f"**Recycling Info:** {recycling_info[class_name]}")
                        st.write(f"**Confidence:** {confidence*100:.2f}%")
                        if confidence >= 0.8:
                            st.info("The model is very confident about this prediction.")
                        elif confidence >= 0.78:
                            st.warning("The model is fairly confident, but there is some uncertainty.")
                        else:
                            st.error("The model is not very confident. The prediction might be unreliable.")
