import numpy as np
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")  # wide works best for top nav

st.markdown("""
    <style>
    /* Neon border around the horizontal menu container */
    .nav-pills {
        border: 2px solid #39ff14;
        border-radius: 12px;
        padding: 5px;
        background-color: black;
        box-shadow: 0 0 5px #39ff14, 0 0 10px #39ff14, 0 0 20px #39ff14;
        transition: 0.3s;
    }

    /* Intensify glow on hover */
    .nav-pills:hover {
        box-shadow: 0 0 10px #39ff14, 0 0 20px #39ff14, 0 0 30px #39ff14, 0 0 50px #39ff14;
    }

    /* Menu item text color */
    .nav-pills .nav-link {
        color: #39ff14 !important;
        font-weight: bold;
    }

    /* Active item gets stronger glow */
    .nav-pills .nav-link.active {
        background-color: #111 !important;
        border: 2px solid #39ff14;
        box-shadow: 0 0 10px #39ff14, 0 0 20px #39ff14, 0 0 30px #39ff14;
    }
    </style>
""", unsafe_allow_html=True)

# --- CSS tweak to move menu to the very top ---
st.markdown(
    """
    <style>
    div.block-container {padding-top: 5rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Navbar at very top ---
selected = option_menu(
    menu_title=None,
    options=["Home", "Data Analysis", "Image Classification"],
    icons=["house", "bar-chart", "gear"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# --- Header/content below navbar ---
st.title('Garbage Image Classification')

if selected == "Home":
    st.write("Build a deep learning model that classifies images of waste into categories like plastic, metal, glass, paper, and organic. This system will assist in automating recycling by sorting garbage based on image input, using a deep learning model deployed via a simple user interface.")
    
    st.markdown(""" ### Business Use Cases:""")
    st.markdown(""" 
    * Smart Recycling Bins: Automatically sort waste into appropriate bins.
    * Municipal Waste Management: Reduce manual sorting time and labor.
    * Educational Tools: Teach proper segregation through visual tools.
    * Environmental Analytics: Track waste composition and recycling trends. 
    """)
    
    st.markdown(""" ### Dataset and Dataset Explanation:""")
    st.markdown(""" 
    * Description: This dataset contains images categorized into six classes: cardboard, glass, metal, paper, plastic, and trash.
    * Size: Approximately 2,467 images.
    * Link: [Kaggle Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification?utm_source=chatgpt.com) 
    """)
    

elif selected == "Data Analysis":
    st.markdown(
    """
    <style>
        /* Reduce padding and center content a bit */
        .block-container {
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
    )

# Layout: left = menu, right = content
    col1, col2 = st.columns([1.2, 4.8], gap="large")  # better proportions + spacing

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
        # Create two inner columns for image and description
            img_col, text_col = st.columns([2, 2], gap = 'large')  

            with img_col:
                st.markdown('<div class="block-container">', unsafe_allow_html=True)
                image = Image.open("EDA Images/no of categories.png")
                st.image(image, caption="Number of images per class", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with text_col:
                st.markdown('<div class="block-container">', unsafe_allow_html=True)
                st.markdown(
                """
                ### 📊 Distribution of Images Across Categories

                - **Paper** has the highest number of images (~600), making it the most represented class.  
                - **Glass** and **Plastic** are also well-represented with around 500 images each.  
                - **Metal** and **Cardboard** have slightly fewer samples (~400 each).  
                - **Trash** has the lowest count (~130), making it a minority class.  
                
                **Key Insight:**  
                The dataset is **imbalanced**, with certain categories (like *trash*) underrepresented compared to others (like *paper*).  
                This imbalance may affect model training and could require techniques such as **data augmentation** or **class weighting** 
                to ensure fair performance across all classes.
                """
                )
                st.markdown('</div>', unsafe_allow_html=True)

        elif selected == "Pixel intensity distribution for each class":
            
            selected = option_menu(
            menu_title=None,
            options=["trash", "plastic", "glass", "metal", "paper", "cardboard"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal"
            )
            img_col, text_col = st.columns([2, 2], gap = 'large')
            
            if selected == "trash":
                with img_col:
                    st.markdown('<div class="block-container">', unsafe_allow_html=True)
                    image = Image.open("EDA Images\Intensity Distribution of pixels\download (1).png")
                    st.image(image, caption="Pixel intensity distribution for trash", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with text_col:
                    st.markdown('<div class="block-container">', unsafe_allow_html=True)
                    st.markdown(
                    """
                    This graph illustrates the pixel intensity distribution for an image of "trash", likely analyzed for image processing or computer vision purposes.
                    1) ***Multiple Lines:*** The red, green, and blue lines likely represent the frequency distribution of pixel intensities for the Red, Green, and Blue color channels of the image, respectively.
                    2) ***Peaks and Valleys:*** The peaks in the graph (e.g., around 170-220 pixel values) suggest a high concentration of pixels with those specific intensity values, indicating dominant brightness levels or colors within the "trash" image, while valleys represent less frequent intensity values.
                    """
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            if selected == "plastic":
                with img_col:
                    st.markdown('<div class="block-container">', unsafe_allow_html=True)
                    image = Image.open("EDA Images\Intensity Distribution of pixels\download (2).png")
                    st.image(image, caption="Pixel intensity distribution for plastic", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with text_col:
                    st.markdown('<div class="block-container">', unsafe_allow_html=True)
                    st.markdown(
                    """
                    The image displays a Pixel Intensity Distribution Histogram for plastic, which is a graphical representation of the tonal or color distribution within an image, revealing insights into its brightness, contrast, and color balance. 
                    Here's an explanation of its components:
                    1) ***Multiple Lines (Red, Green, Blue):*** The red, green, and blue lines likely represent the frequency distribution of pixel intensities for the Red, Green, and Blue color channels of the image, respectively.
                    2) ***Peaks and Valleys:*** Peaks (high points in the graph) signify pixel intensity values that occur frequently in the image, indicating dominant tones or brightness levels.
                    """
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            if selected == "metal":
                with img_col:
                    st.markdown('<div class="block-container">', unsafe_allow_html=True)
                    image = Image.open("EDA Images\Intensity Distribution of pixels\download (4).png")
                    st.image(image, caption="Pixel intensity distribution for metal", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with text_col:
                    st.markdown('<div class="block-container">', unsafe_allow_html=True)
                    st.markdown(
                    """
                    The image you provided is a color histogram, which graphically represents the distribution of pixel intensities within an image. It's a fundamental tool in digital image processing for analyzing the tonal and color composition of an image. 
                    Here's an explanation of its components:
                    1) ***Multiple Lines (Red, Green, Blue):*** In a color image like the one suggested by your histogram, each line (Red, Green, Blue) represents the intensity distribution for that specific color channel.
                    2) ***Peaks and Valleys:*** Peaks indicate a high concentration of pixels at those specific intensity levels.
                    """
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    
            if selected == "glass":
                with img_col:
                    st.markdown('<div class="block-container">', unsafe_allow_html=True)
                    image = Image.open("EDA Images\Intensity Distribution of pixels\download (3).png")
                    st.image(image, caption="Pixel intensity distribution for glass", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with text_col:
                    st.markdown('<div class="block-container">', unsafe_allow_html=True)
                    st.markdown(
                    """
                    This graph illustrates the Pixel Intensity Distribution for Glass, also known as an RGB Histogram, providing a visual representation of the tonal and color distribution of an image of glass. 
                    1) ***Multiple Lines (Red, Green, Blue):*** Each colored line (red, green, blue) corresponds to the individual color channel histogram of the image. For an RGB image, separate histograms are generated for the red, green, and blue components, showing the intensity distribution for each primary color across all pixels.
                    2) ***Peaks and Valleys:*** Peaks, These indicate the pixel intensity values that are most frequently present in the image for a given color channel.
                    """
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            if selected == "paper":
                with img_col:
                    st.markdown('<div class="block-container">', unsafe_allow_html=True)
                    image = Image.open("EDA Images\Intensity Distribution of pixels\download (5).png")
                    st.image(image, caption="Pixel intensity distribution for metal", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with text_col:
                    st.markdown('<div class="block-container">', unsafe_allow_html=True)
                    st.markdown(
                    """
                    The image you provided is a Pixel Intensity Distribution Histogram for a paper image, commonly used in image processing to visualize the tonal distribution and color characteristics of an image. 
                    Here's a breakdown of its components:
                    1) ***Multiple Lines (Red, Green, Blue):*** These lines represent the histograms of the individual color channels (Red, Green, and Blue) of the image, as digital color images are often composed of these three primary additive color channels.
                    2) ***Peaks and Valleys:*** peaks, These features in the histogram provide insights into the image's tonal characteristics. Represent pixel intensity values that are highly frequent in the image.
                    """
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            if selected == "cardboard":
                with img_col:
                    st.markdown('<div class="block-container">', unsafe_allow_html=True)
                    image = Image.open("EDA Images\Intensity Distribution of pixels\download (6).png")
                    st.image(image, caption="Pixel intensity distribution for cardboard", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with text_col:
                    st.markdown('<div class="block-container">', unsafe_allow_html=True)
                    st.markdown(
                    """
                    This image displays a Pixel Intensity Distribution Histogram for "cardboard", which is a graphical representation of the tonal distribution in a digital image. Here's an explanation of its components:
                    
                    1) ***Multiple Lines (Red, Green, Blue):*** These lines represent the histograms of the individual color channels (Red, Green, and Blue) of the image, as digital color images are often composed of these three primary additive color channels.
                    2) ***Peaks and Valleys:*** Peaks in the histogram (high points on the lines) indicate that a large number of pixels in the image have the corresponding pixel intensity value on the X-axis. Valleys (low points) suggest that fewer pixels have those specific intensity values.
                    """
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
            

        elif selected == "Example Images":
            selected = option_menu(
            menu_title=None,
            options=["trash", "plastic", "glass", "metal", "paper", "cardboard"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            )
            
            if selected == "trash":
                col1, col2, col3 = st.columns([1,2,1])  # center the image in middle column
                with col2:
                    image = Image.open("EDA Images\Example Images\Screenshot 2025-08-25 211343.png")
                    st.image(image, caption=f"Pixel intensity distribution for {selected}", use_container_width=True)

                
            if selected == "plastic":
                col1, col2, col3 = st.columns([1,2,1])  # center the image in middle column
                with col2:
                    image = Image.open("EDA Images\Example Images\Screenshot 2025-08-25 211359.png")
                    st.image(image, caption=f"Example image for {selected}", use_container_width=True)
                
            if selected == "glass":
                col1, col2, col3 = st.columns([1,2,1])  # center the image in middle column
                with col2:
                    image = Image.open("EDA Images\Example Images\Screenshot 2025-08-25 211443.png")
                    st.image(image, caption=f"Example image for {selected}", use_container_width=True)
                
            if selected == "metal":
                col1, col2, col3 = st.columns([1,2,1])  # center the image in middle column
                with col2:
                    image = Image.open("EDA Images\Example Images\Screenshot 2025-08-25 211459.png")
                    st.image(image, caption=f"Example image for {selected}", use_container_width=True)
                
            if selected == "paper":
                col1, col2, col3 = st.columns([1,2,1])  # center the image in middle column
                with col2:
                    image = Image.open("EDA Images\Example Images\Screenshot 2025-08-25 211518.png")
                    st.image(image, caption=f"Example image for {selected}", use_container_width=True)
                
            if selected == "cardboard":
                col1, col2, col3 = st.columns([1,2,1])  # center the image in middle column
                with col2:
                    image = Image.open("EDA Images\Example Images\Screenshot 2025-08-25 211538.png")
                    st.image(image, caption=f"Example image for {selected}", use_container_width=True)
                        
elif selected == "Image Classification":
        
        col1, col2 = st.columns([1.2, 4.8], gap="large")  # better proportions + spacing
    
        with col1:
            with st.container():
                st.markdown('<div class="block-container">', unsafe_allow_html=True)
                selected = option_menu(
                menu_title="Main Menu",
                options=["Image Classification", 
                         "Model performances", 
                         ],
                icons=["bi-cpu", "graph-up"],
                menu_icon="cast",
                default_index=0,
                orientation="vertical"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
        with col2:
            if selected == "Image Classification":
                # Load your trained .h5 model
                model = load_model("InceptionV3_final.h5")
    
                # Define class names in the same order as training
                dic = {0 : "cardboard", 3: "Paper", 1: "Glass", 2: "Metal", 4: "Plastic", 5: "Trash"}
                recycling_info = {
                "cardboard": "Can be recycled in most curbside programs. Remove any tape or non-paper materials.",
                "Glass": "Rinse and recycle in designated glass bins. Colored glass may have restrictions.",
                "Metal": "Clean and place in metal recycling bins. Includes aluminum cans and tin containers.",
                "Paper": "Recycle paper products. Avoid greasy or heavily soiled paper.",
                "Plastic": "Rinse and recycle plastics labeled 1-7 where accepted. Avoid bags in curbside bins.",
                "Trash": "Non-recyclable waste. Dispose of in regular trash bins."
                }

                # Upload image
                uploaded_file = st.file_uploader("choose the image:", type=["jpg", "jpeg", "png"])
    
                if uploaded_file is not None:
                    img = Image.open(uploaded_file).convert("RGB")
        
                    # Use columns for layout
                    col1, col2 = st.columns(2, gap="large")
        
                    with col1:
                        st.image(img, caption="Uploaded Image", use_container_width=True)
        
                        # Preprocess image
                        img_resized = img.resize((224, 224))
                        img_array = np.array(img_resized) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
        
                        # Make prediction
                        predictions = model.predict(img_array)
                        predicted_class = np.argmax(predictions[0])
                        confidence = np.max(predictions[0])
                        class_name = dic[predicted_class]
        
                    with col2:
                        st.subheader("Prediction Details")
                        st.write(f"**Class:** {class_name}")
                        st.write(f"**Recycling Info:** {recycling_info[class_name]}")
                        st.write(f"**Confidence:** {confidence*100:.2f}%")
                        if confidence >= 0.9:
                            st.info("The model is very confident about this prediction.")
                        elif confidence >= 0.7:
                            st.warning("The model is fairly confident, but there is some uncertainty.")
                        else:
                            st.error("The model is not very confident. The prediction might be unreliable.")
                            
            elif selected == "Model performances":
                    
                selected = option_menu(
                menu_title=None,
                options=["EfficientNet", "ResNet", "MobileNet", "DenseNet", "InceptionNet", "Performance comparison"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal",
                )
                
                all_metrics = {
                "EfficientNet": {
                    'precision': {'cardboard': 0.00, 'glass': 0.20, 'metal': 0.00, 'paper': 0.00, 'plastic': 0.00, 'trash': 0.00, 'accuracy': 0.20, 'macro avg': 0.03, 'weighted avg': 0.04},
                    'recall': {'cardboard': 0.00, 'glass': 1.00, 'metal': 0.00, 'paper': 0.00, 'plastic': 0.00, 'trash': 0.00, 'accuracy': 0.20, 'macro avg': 0.17, 'weighted avg': 0.20},
                    'f1-score': {'cardboard': 0.00, 'glass': 0.33, 'metal': 0.00, 'paper': 0.00, 'plastic': 0.00, 'trash': 0.00, 'accuracy': 0.20, 'macro avg': 0.06, 'weighted avg': 0.07},
                    'support': {'cardboard': 80, 'glass': 100, 'metal': 82, 'paper': 118, 'plastic': 96, 'trash': 27, 'accuracy': 503, 'macro avg': 503, 'weighted avg': 503}
                },
                "ResNet": {
                    'precision': {'cardboard': 0.00, 'glass': 0.00, 'metal': 0.43, 'paper': 0.58, 'plastic': 0.24, 'trash': 0.00, 'accuracy': 0.30, 'macro avg': 0.21, 'weighted avg': 0.25},
                    'recall': {'cardboard': 0.00, 'glass': 0.00, 'metal': 0.12, 'paper': 0.42, 'plastic': 0.97, 'trash': 0.00, 'accuracy': 0.30, 'macro avg': 0.25, 'weighted avg': 0.30},
                    'f1-score': {'cardboard': 0.00, 'glass': 0.00, 'metal': 0.19, 'paper': 0.49, 'plastic': 0.38, 'trash': 0.00, 'accuracy': 0.30, 'macro avg': 0.18, 'weighted avg': 0.22},
                    'support': {'cardboard': 80, 'glass': 100, 'metal': 82, 'paper': 118, 'plastic': 96, 'trash': 27, 'accuracy': 503, 'macro avg': 503, 'weighted avg': 503}
                },
                "MobileNet": {
                    'precision': {'cardboard': 0.92, 'glass': 0.65, 'metal': 0.75, 'paper': 0.75, 'plastic': 0.66, 'trash': 0.59, 'accuracy': 0.73, 'macro avg': 0.72, 'weighted avg': 0.73},
                    'recall': {'cardboard': 0.75, 'glass': 0.70, 'metal': 0.78, 'paper': 0.91, 'plastic': 0.59, 'trash': 0.37, 'accuracy': 0.73, 'macro avg': 0.68, 'weighted avg': 0.73},
                    'f1-score': {'cardboard': 0.83, 'glass': 0.68, 'metal': 0.77, 'paper': 0.82, 'plastic': 0.62, 'trash': 0.45, 'accuracy': 0.73, 'macro avg': 0.70, 'weighted avg': 0.73},
                    'support': {'cardboard': 80, 'glass': 100, 'metal': 82, 'paper': 118, 'plastic': 96, 'trash': 27, 'accuracy': 503, 'macro avg': 503, 'weighted avg': 503}
                },
                "DenseNet": {
                    'precision': {'cardboard': 0.85, 'glass': 0.61, 'metal': 0.70, 'paper': 0.78, 'plastic': 0.66, 'trash': 0.45, 'accuracy': 0.70, 'macro avg': 0.67, 'weighted avg': 0.70},
                    'recall': {'cardboard': 0.71, 'glass': 0.69, 'metal': 0.68, 'paper': 0.91, 'plastic': 0.58, 'trash': 0.33, 'accuracy': 0.70, 'macro avg': 0.65, 'weighted avg': 0.70},
                    'f1-score': {'cardboard': 0.78, 'glass': 0.64, 'metal': 0.69, 'paper': 0.84, 'plastic': 0.66, 'trash': 0.38, 'accuracy': 0.70, 'macro avg': 0.66, 'weighted avg': 0.70},
                    'support': {'cardboard': 80, 'glass': 100, 'metal': 82, 'paper': 118, 'plastic': 96, 'trash': 27, 'accuracy': 503, 'macro avg': 503, 'weighted avg': 503}
                },
                "InceptionNet": {
                    'precision': {'cardboard': 0.94, 'glass': 0.77, 'metal': 0.68, 'paper': 0.75, 'plastic': 0.81, 'trash': 0.64, 'accuracy': 0.77, 'macro avg': 0.77, 'weighted avg': 0.78},
                    'recall': {'cardboard': 0.76, 'glass': 0.77, 'metal': 0.82, 'paper': 0.92, 'plastic': 0.69, 'trash': 0.33, 'accuracy': 0.77, 'macro avg': 0.72, 'weighted avg': 0.77},
                    'f1-score': {'cardboard': 0.84, 'glass': 0.77, 'metal': 0.74, 'paper': 0.83, 'plastic': 0.75, 'trash': 0.44, 'accuracy': 0.77, 'macro avg': 0.73, 'weighted avg': 0.77},
                    'support': {'cardboard': 80, 'glass': 100, 'metal': 82, 'paper': 118, 'plastic': 96, 'trash': 27, 'accuracy': 503, 'macro avg': 503, 'weighted avg': 503}
                }
            }
                
                if selected == "Performance comparison":
                    import pandas as pd
                    import plotly.express as px

                    
                    st.title("Model Performance Visualization")

                        # Data from your Colab output
                    validation_accuracies = {
                    'ResNet50': 0.3042,
                    'EfficientNetB0': 0.1988,
                    'MobileNetV2': 0.7316,
                    'DenseNet121': 0.7038,
                    'InceptionV3': 0.7734
                    }

                    # Create a DataFrame for plotting
                    df = pd.DataFrame(
                    validation_accuracies.items(),
                    columns=['Model', 'Validation Accuracy']
                    )
                    # Create the bar chart with Plotly Express
                    fig = px.bar(
                        df,
                        x='Model',
                        y='Validation Accuracy',
                        title='Model Comparison by Validation Accuracy',
                        color='Model',
                        text='Validation Accuracy',
                        hover_data={'Validation Accuracy': ':.2%'}
                    )
                    # Customize the chart layout
                    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                    fig.update_layout(
                        xaxis_title="Model",
                        yaxis_title="Validation Accuracy",
                        yaxis_tickformat=".0%",
                        uniformtext_minsize=8,
                        uniformtext_mode='hide'
                    )
                    # Display the chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif selected in all_metrics:

                    model_f1_scores = {
                    "ResNet": {
                    "Validation Accuracy": "30.42%",
                    "Performance": "ResNet50 performed poorly on this specific dataset, with an overall accuracy of just over 30%.",
                    "Key Strength": "The model's best performance was on the plastic and paper classes. It had high recall (97%) for plastic, meaning it correctly identified almost all actual plastic items, though its low precision (24%) indicates it also misclassified many non-plastic items as plastic. For paper, both its precision (58%) and recall (42%) were its highest, but still relatively low.",
                    "Areas for Improvement": "The model failed to predict several classes, including cardboard, glass, and trash, for which it recorded a precision and recall of 0.00.",
                    },

                    "EfficientNet": {
                    "Validation Accuracy": "19.88%",
                    "Performance": "This model was the worst performer of the group, with a validation accuracy under 20%.",
                    "Key Strength": "Its only successful predictions came from the glass class, for which it achieved a perfect recall (1.00). However, its precision of 0.20 suggests it also incorrectly identified many other items as glass.",
                    "Areas for Improvement": "EfficientNetB0 completely failed to identify any instances of cardboard, metal, paper, plastic, and trash in the test set.",
                    },

                    "MobileNet": {
                    "Validation Accuracy": "73.16%",
                    "Performance": "This is a strong, balanced model for your task, with an overall accuracy of over 73%. It is specifically designed for mobile and embedded devices, offering high performance with a low computational footprint.",
                    "Key Strength": "MobileNetV2 performed well across most classes, particularly on cardboard (precision 92%, recall 75%) and paper (precision 75%, recall 91%).",
                    "Areas for Improvement": "Performance on the trash class was less robust (precision 59%, recall 37%).",
                    },

                    "DenseNet": {
                    "Performance": "DenseNet121 delivered a solid performance, with a validation accuracy of over 70%.",
                    "Key Strength": "The model was particularly effective at classifying paper (recall 91%, precision 78%) and showed strong results for cardboard (precision 85%, recall 71%).",
                    "Areas for Improvement": "Similar to MobileNetV2, the performance on the trash category was the weakest.",
                    "Validation Accuracy": "70.38%",
                    },

                    "InceptionNet": {
                    "Performance": "InceptionV3 is the best-performing model for this dataset, with the highest validation accuracy at over 77%.",
                    "Key Strength": "It demonstrated excellent performance across multiple categories, with high f1-scores for cardboard, glass, metal, and paper. Its predictions for paper were particularly accurate (precision 75%, recall 92%).",
                    "Areas for Improvement": "It had the lowest recall on the trash class, though its precision was moderate.",
                    "Validation Accuracy": "77.34%"
                    }
                }
                
                
                

                    # Top row with two boxes    
                    col1, col2 = st.columns([2, 2], gap="large")

                    with col1:
                        with st.container():
                            metrics_data = all_metrics[selected]
                            df_metrics = pd.DataFrame(metrics_data)
                            st.subheader(f"{selected} Evaluation Metrics")
                            st.dataframe(df_metrics, use_container_width=True)

                    with col2:
                        with st.container():
                            st.subheader(f"{selected} Description")
                            st.metric("Validation Accuracy:", f"{model_f1_scores[selected]['Validation Accuracy']}")
                            st.write(f"{model_f1_scores[selected]['Performance']}")
                            st.write(f"{model_f1_scores[selected]['Key Strength']}")
                            st.write(f"{model_f1_scores[selected]['Areas for Improvement']}")
                        # Bottom row with one large box
                    with st.container():
                        with st.expander("Click to see the confusion matrix"):
                            if selected == 'EfficientNet':
                                image = Image.open("confusion matrix/efficientnet.png")
                                st.image(image, caption=f"{selected} confusion matrix", use_container_width=True)
                            if selected == 'InceptionNet':
                                image = Image.open("confusion matrix/inception.png")
                                st.image(image, caption=f"{selected} confusion matrix", use_container_width=True)
                            if selected == 'DenseNet':
                                image = Image.open("confusion matrix/densenet.png")
                                st.image(image, caption=f"{selected} confusion matrix", use_container_width=True)
                            if selected == 'MobileNet':
                                image = Image.open("confusion matrix/mobilenet.png")
                                st.image(image, caption=f"{selected} confusion matrix", use_container_width=True)
                            if selected == 'ResNet':
                                image = Image.open("confusion matrix/resnet.png")
                                st.image(image, caption=f"{selected} confusion matrix", use_container_width=True)

