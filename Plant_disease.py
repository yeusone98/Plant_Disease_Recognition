import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit.web.cli import main

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    nav {
        background-color: #2e7bcf;
        padding: 10px 0;
        width: 100%;
        position: fixed;
        top: 0;
        left: 0;
        z-index: 1000;
    }
    .nav-wrapper {
        display: flex;
        justify-content: flex-end;
        padding: 0 20px;
    }
    #nav-mobile {
        list-style-type: none;
        margin: 0;
        padding: 0;
        display: flex;
    }
    .nav-link {
        color: white;
        text-decoration: none;
        padding: 10px 15px;
        transition: background-color 0.3s;
    }
    .nav-link:hover {
        background-color: #3e8bd0;
    }
    .main-content {
        padding-top: 60px;
    }
    .stButton>button {
        color: #fff;
        background-color: #4CAF50;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stFileUploader>div {
        border: 2px dashed #4CAF50;
        padding: 20px;
        border-radius: 10px;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stFileUploader>div:hover {
        border-color: #45a049;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .uploaded-image {
        border-radius: 10px;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        padding: 20px;
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #2e7bcf;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Horizontal menu
st.markdown("""
    <nav>
        <div class="nav-wrapper">
            <ul id="nav-mobile" class="right">
                <li><a href="/" class="nav-link">Home</a></li>
                <li><a href="/About" class="nav-link">About</a></li>
                <li><a href="/Disease_Recognition" class="nav-link">Disease Recognition</a></li>
            </ul>
        </div>
    </nav>
""", unsafe_allow_html=True)

# Get current page from URL
app_mode = st.experimental_get_query_params().get("page", ["Home"])[0]

# Main content
st.markdown('<div class="main-content">', unsafe_allow_html=True)

if app_mode == "Home":
    st.title("🌱 Plant Disease Recognition System 🌿")
    st.image("image-20-edited.jpg", use_column_width=True, caption="Healthy Plant")
    st.write("""
    ## Welcome!
    Our goal is to help identify plant diseases quickly and accurately. Simply upload an image of a plant, and our system will analyze it to detect any signs of disease. Let's work together to protect our crops and ensure a healthy harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected disease.
    2. **Analysis:** Our system processes the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** We use state-of-the-art machine learning techniques for precise disease detection.
    - **User-Friendly:** Simple and intuitive interface for a seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, enabling quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the menu to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "About":
    st.title("About the Project")
    st.write("""
    #### About Dataset
    This dataset was recreated using offline augmentation from the original dataset. The original dataset is available on this GitHub repository.
    It contains approximately 87,000 RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The dataset is split into a 80/20 ratio for training and validation sets, preserving the directory structure. A new directory with 33 test images was created later for prediction purposes.

    #### Content
    - **Train:** 70,295 images
    - **Test:** 33 images
    - **Validation:** 17,572 images
    """)

elif app_mode == "Disease_Recognition":
    st.title("Disease Recognition")
    test_image = st.file_uploader("Upload an Image:", type=["jpg", "png", "jpeg"])
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict"):
            with st.spinner("Analyzing..."):
                result_index = model_prediction(test_image)
                class_names = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                st.success(f"Model predicts: {class_names[result_index]}")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>&copy; 2024 Plant Disease Recognition System. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)

# JavaScript for handling page navigation
st.markdown("""
    <script>
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = link.getAttribute('href').replace('/', '');
            window.history.pushState({}, '', `?page=${page}`);
            window.location.reload();
        });
    });
    </script>
""", unsafe_allow_html=True)