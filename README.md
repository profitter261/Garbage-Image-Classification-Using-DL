# Garbage Image Classification Using Deep Learning:-
I have Built a deep learning model that classifies images of waste into categories like plastic, metal, glass, paper, and organic. This system will assist in automating recycling by sorting garbage based on image input, using a deep learning model deployed via a simple user interface.

## Business Use Cases:-
- ***Smart Recycling Bins***: Automatically sort waste into appropriate bins.
- ***Municipal Waste Management***: Reduce manual sorting time and labor.
- ***Educational Tools***: Teach proper segregation through visual tools.
- ***Environmental Analytics***: Track waste composition and recycling trends.

## Approach:-
- ***Data Preparation***
   - Garbage images categorized by type (plastic, metal, glass, paper, etc.).
- ***Data Cleaning & Preprocessing***
   - Scripts for image resizing, normalization, and augmentation.
- ***Exploratory Data Analysis (EDA)***
   - Visualize number of images per class, show example images from each category, analyze pixel intensity or color distribution.
- ***Model Development and model selection***
   - Baseline CNN and Transfer Learning models (MobileNetV2, EfficientNetB0, etc.).
- ***Model Evaluation***
   - Performance comparison table using metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
- ***Application Development***
   - Interactive UI to upload and classify images with prediction labels.

## Results:-
- ### Cardboard Garbage Classification:-
  - ![Cardboard Garbage Classification](https://github.com/profitter261/Garbage-Image-Classification-Using-DL/blob/main/Prediction%20Images/Cardboard%20Garbage%20Prediction.png)
- ### Glass Garbage Classification:-
   - ![Glass Garbage Classification](https://github.com/profitter261/Garbage-Image-Classification-Using-DL/blob/main/Prediction%20Images/Garbage%20Glass%20Prediction.png)
- ### Metal Garbage Classification:-
   - ![Metal Garbage Classification](https://github.com/profitter261/Garbage-Image-Classification-Using-DL/blob/main/Prediction%20Images/Garbage%20Metal%20Prediction.png)
- ### Paper Garbage Classification:-
   - ![Paper Garbage Classification](https://github.com/profitter261/Garbage-Image-Classification-Using-DL/blob/main/Prediction%20Images/Paper%20Garbage%20Prediction.png)
- ### Plastic Garbage Classification:-
   - ![Cardboard Garbage Classification](https://github.com/profitter261/Garbage-Image-Classification-Using-DL/blob/main/Prediction%20Images/Plastic%20Garbage%20Prediction.png)
- ### Trash Garbage Classification:-
   - ![Trash Garbage Classification](https://github.com/profitter261/Garbage-Image-Classification-Using-DL/blob/main/Prediction%20Images/Trash%20Class%20Prediction.png)

## Results:- 
- Preprocessed, augmented dataset
- Multiple deep learning models trained and evaluated
- Final app with above 80% accuracy waste category prediction
- Accessible UI using Streamlit

## Conclusion:-
This project demonstrated the application of Deep Learning for waste classification, enabling the automatic categorization of garbage images into predefined classes (e.g., recyclable, organic, hazardous, etc.). By leveraging convolutional neural networks (CNNs), the model achieved strong classification performance, showing that AI can significantly aid in waste management and recycling processes.
Such a system can be deployed in smart bins, waste segregation units, or recycling plants, where it helps:

- Reduce human effort and error in manual sorting.
- Improve recycling efficiency by separating materials correctly.
- Contribute to sustainable waste management practices.

While the model performs well on test data, real-world deployment may require further improvements such as handling noisy/unclean images, class imbalance, and scaling to larger datasets. Future work could also involve edge deployment on IoT devices or integrating with computer vision systems in waste processing plants.
In conclusion, this project highlights how AI and deep learning can make a tangible impact on environmental sustainability, moving us closer to smarter and cleaner cities
  
## Dataset:-
- ### Garbage Classification (6 Classes):-
    - ***Description:*** This dataset contains images categorized into six classes: cardboard, glass, metal, paper, plastic, and trash.
    - ***Size:*** Approximately 2,467 images.
    - ***Link:***:-  [Kaggle Dataset for Garbage Classification (6 Classes)](https://www.kaggle.com/datasets/asdasdasasdas/garbageclassification?utm_source=chatgpt.com)

