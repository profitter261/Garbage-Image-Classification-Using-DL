# Import necessary libraries
import numpy as np  # For numerical computations
import streamlit as st  # For creating the web app
import pandas as pd  # For data manipulation
from tensorflow.keras.models import load_model  # For loading trained Keras models
from tensorflow.keras.preprocessing import image  # For image preprocessing
import cv2  # For image processing
from PIL import Image  # For handling images
import plotly.express as px  # For interactive plots
import plotly.graph_objects as go  # For advanced plotting
from streamlit_option_menu import option_menu  # For creating navigation menus
from streamlit_lottie import st_lottie  # For embedding Lottie animations
import requests  # For fetching animations from URLs
import json  # For loading JSON Lottie files
import time  # For adding delays
import gdown

MODEL_PATH = "InceptionV3_final.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=17tOiPzn4l-5uhvg1PETRkZ5-3YOMG4Vi"

# Download if not present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... This may take a minute ⏳")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# Load model
model = load_model(MODEL_PATH)
    
# Set page layout to wide for better UI
st.set_page_config(layout="wide")

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
# Load local Lottie animation for trash concept
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
st.markdown(
    """
    <style>
    div.block-container {padding-top: 5rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Main navigation menu using streamlit_option_menu ---
selected = option_menu(
    menu_title=None,
    options=["Home", "Data Analysis", "Image Classification"],
    icons=["house", "bar-chart", "gear"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# --- Header of the web app ---
st.title('Garbage Image Classifier')

# ------------------------- HOME PAGE -------------------------
if selected == "Home":
    left_col, right_col = st.columns([2, 1])  # Split page into text and animation

    with left_col:
        # Project description
        st.write("A garbage image classifier that can automatically identify and categorize waste into types such as plastic, metal, glass, paper, and organic. By leveraging advanced image recognition techniques, the system provides an efficient way to sort garbage, supporting more streamlined recycling processes. The model is deployed through a user-friendly Streamlit interface, allowing users to simply upload an image of waste and receive instant classification results, making waste management smarter and more accessible.")
        
        # Business use cases
        st.markdown("### Business Use Cases:")
        st.markdown(""" 
        * Smart Recycling Bins: Automatically sort waste into appropriate bins.
        * Municipal Waste Management: Reduce manual sorting time and labor.
        * Educational Tools: Teach proper segregation through visual tools.
        * Environmental Analytics: Track waste composition and recycling trends. 
        """)
        
        # Dataset information
        st.markdown("### Dataset and Dataset Explanation:")
        st.markdown(""" 
        * Description: This dataset contains images categorized into six classes: cardboard, glass, metal, paper, plastic, and trash.
        * Size: Approximately 2,467 images.
        * Link: [Kaggle Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
        """)

    # Right column shows Lottie animation
    with right_col:
        st_lottie(trash_lottie, speed=1, loop=True, height=300)

# --------------------- DATA ANALYSIS PAGE ---------------------
elif selected == "Data Analysis":
    # Adjust layout for content
    st.markdown(
        """
        <style>
            .block-container {
                padding-bottom: 1rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Split layout: left = sub-menu, right = content
    col1, col2 = st.columns([1.2, 4.8], gap="large")

    with col1:
        with st.container():
            st.markdown('<div class="block-container">', unsafe_allow_html=True)
            # Sub-menu for different EDA visualizations
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
            # Columns for chart and explanation
            img_col, text_col = st.columns([2, 2], gap='large')  

            with img_col:
                # Sample dataset counts
                categories = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
                counts = [403, 501, 410, 594, 482, 137]

                # Plot number of images per class
                fig = px.bar(
                    x=categories, 
                    y=counts, 
                    title="📊 Number of Images Per Class",
                    text=counts,
                    color=categories,
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    xaxis_title="Class",
                    yaxis_title="Number of Images",
                    title_x=0.2,
                    template="plotly_dark"
                )

                st.plotly_chart(fig, use_container_width=True)
                
            with text_col:
                st.markdown(
                    """
                    ### 📊 Distribution of Images Across Categories

                    - **Paper** has the highest number of images (~600), making it the most represented class.  
                    - **Glass** and **Plastic** are also well-represented with around 500 images each.  
                    - **Metal** and **Cardboard** have slightly fewer samples (~400 each).  
                    - **Trash** has the lowest count (~130), making it a minority class.  
                    """
                )
                st.markdown(
                    "**Key Insight:** The dataset is **imbalanced**, which may require **data augmentation** or **class weighting** to ensure fair model performance across all classes."
                )

        elif selected == "Pixel intensity distribution for each class":
            # Horizontal sub-menu for class selection
            selected = option_menu(
                menu_title=None,
                options=["trash", "plastic", "glass", "metal", "paper", "cardboard"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal"
            )
            img_col, text_col = st.columns([2, 2], gap='large')

            # Loop through classes for pixel intensity plotting
            img_path = f"EDA Images\\Example Images\\{selected.capitalize()}.png"
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Calculate histogram for each RGB channel
            hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256]).flatten()
            hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256]).flatten()
            hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256]).flatten()

            # Plot pixel intensity distributions
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(256), y=hist_r, mode='lines', name='Red', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=np.arange(256), y=hist_g, mode='lines', name='Green', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=np.arange(256), y=hist_b, mode='lines', name='Blue', line=dict(color='blue')))
            fig.update_layout(title=f"Pixel intensity distribution for {selected.capitalize()} Images",
                              xaxis_title="Pixel value", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

            # Explanation text
            st.markdown(
                f"This graph shows the distribution of pixel intensities for the '{selected}' image across Red, Green, and Blue channels. Peaks represent common intensity values in the image, while valleys indicate less frequent intensities."
            )

        elif selected == "Example Images":
            # Horizontal sub-menu for example images
            selected = option_menu(
                menu_title=None,
                options=["trash", "plastic", "glass", "metal", "paper", "cardboard"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal",
            )
            col1, col2, col3 = st.columns([1, 2, 1])  # Center image
            with col2:
                image = Image.open(f"EDA Images\\Example Images\\{selected.capitalize()}.png")
                st.image(image, caption=f"Example image for {selected}", use_container_width=True)

# ------------------- IMAGE CLASSIFICATION PAGE -------------------
elif selected == "Image Classification":
    col1, col2 = st.columns([1.2, 4.8], gap="large")

    with col1:
        with st.container():
            st.markdown('<div class="block-container">', unsafe_allow_html=True)
            # Vertical sub-menu for classification tasks
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
            # Load trained model
            

            # Class mapping dictionary (must match training)
            dic = {0 : "cardboard", 3: "Paper", 1: "Glass", 2: "Metal", 4: "Plastic", 5: "Trash"}

            # Recycling instructions for each class
            recycling_info = {
                "cardboard": "Can be recycled in most curbside programs. Remove any tape or non-paper materials.",
                "Glass": "Rinse and recycle in designated glass bins. Colored glass may have restrictions.",
                "Metal": "Clean and place in metal recycling bins. Includes aluminum cans and tin containers.",
                "Paper": "Recycle paper products. Avoid greasy or heavily soiled paper.",
                "Plastic": "Rinse and recycle plastics labeled 1-7 where accepted. Avoid bags in curbside bins.",
                "Trash": "Non-recyclable waste. Dispose of in regular trash bins."
            }

            # Load Lottie animation for processing
            with open("src/Loading.json", "r", encoding="utf-8") as f:
                loading_lottie = json.load(f)

            uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert("RGB")

                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.image(img, caption="Uploaded Image", use_container_width=True)

                with col2:
                    placeholder = st.empty()
                    # Show loading animation
                    with placeholder.container():
                        st_lottie(loading_lottie, speed=1, loop=True, height=200, key="loading")
                        st.markdown("<p style='text-align:center;'>Analyzing image... please wait ⏳</p>", unsafe_allow_html=True)

                # Preprocess image for model
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                time.sleep(1.5)  # Optional delay to show animation

                # Predict
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                class_name = dic[predicted_class]

                # Show prediction results
                with col2:
                    placeholder.empty()  # clear animation
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




