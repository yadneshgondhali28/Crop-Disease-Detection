import streamlit as st
import tensorflow as tf
import numpy as np


def model_predict(test_img):
  model = tf.keras.models.load_model('./crop_disease_model.h5')
  image = tf.keras.preprocessing.image.load_img(test_img, target_size=(128,128))
  input_arr = tf.keras.preprocessing.image.img_to_array(image)
  input_arr = np.array([input_arr]) 
  prediction = model.predict(input_arr)
  result_index = np.argmax(prediction)
  return result_index

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Go to", ["Home", "About", "Detect Plant Disease"])

if app_mode == "Home":
  st.header("Crop Disease Detection System")
  img_path = "./image_for_app.jpg"
  st.image(img_path, use_column_width=True)
  st.markdown("""
    ### Welcome to the Crop Disease Detection App! 👋
    This app is designed to help farmers, researchers, and agricultural enthusiasts detect crop diseases from images of leaves using cutting-edge machine learning models.
    With a simple image upload, our app can predict whether the crop leaf is healthy or affected by a disease.

    ---

    ### How to Use the App:
    1. **Prepare the Image**: Ensure you have a clear image of the crop leaf you want to analyze. Supported formats include `.jpg` and `.png`.
    2. **Upload the Image**: Use the file uploader below to upload the image of the crop leaf.
    3. **Analyze the Result**: Once the image is uploaded, the app will process it and predict the disease (or show if the leaf is healthy).
    4. **Get Insights**: You'll see the class prediction of the leaf's condition, helping you take further action in managing your crops.
              
    ---
    
    ### Get Started: 
    This app supports a range of common crops and their diseases. Upload an image to get started! by going to "Detect Plant Disease" page.

    ---
    
    ### About: 
    Learn more about the project in the "About" page.
    """
  )
elif app_mode == "About":
  st.header("About Crop Disease Detection App")
  st.markdown("""
    ### About Dataset:
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo. This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.
              
    ### Content:
    1. Train (70295 images)
    2. Valid (17572 images)
    3. Test (33 images)
              
    ### About the App:
    Our app leverages a machine learning model trained on a large dataset of crop diseases. 
    The goal of this app is to:
    - Help identify common diseases affecting crops such as [add crop names here, e.g., corn, wheat, tomato, etc.].
    - Provide insights that allow users to take action to prevent crop damage and improve yields.

    This tool is especially useful for:
    - **Farmers**: Get quick insights on your crops and take preventive measures to mitigate losses.
    - **Researchers**: Use the tool as a reference or as a supplemental tool in your research.
    - **Agriculture Enthusiasts**: Learn about the diseases affecting different crops and how AI can assist in agriculture.

    ---

    The model currently supports detecting diseases in the following crops:
    - [Potato] - Diseases: [Potato Early Blight]
    - [Tomato] - Diseases: [Tomato Early Blight]
    - [Apple]  - Diseases: [Cedar Apple Rust]
    - [Corn]   - Diseases: [Cercospora Leaf Spot, Common Rust, Northern Leaf Blight]
    - [Grape]   - Diseases: [Black rot, Esca (Black Measles), Leaf Blight]
    - [Peach]   - Diseases: [Bacterial_spot]

    ### Why It Matters:
    Agricultural productivity is key to feeding the growing global population. Early detection of crop diseases helps prevent widespread damage, reduces economic losses, and ensures better crop yield and food security.

    ---

    Thank you for using the **Crop Disease Detection App**! If you have any feedback, feel free to reach out. 🚀
  """)

elif app_mode == "Detect Plant Disease":
  st.header("Disease Recognition")
  test_img = st.file_uploader("Choose an image")
  if st.button("show image"):
    st.image(test_img, use_column_width=True)

  if st.button("Predict"):
    st.write("Our prediction")
    result_index = model_predict(test_img)
    class_names = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust', 
                   'Apple___healthy',
                    'Blueberry___healthy', 
                    'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 
                    'Corn_(maize)___Northern_Leaf_Blight', 
                    'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 
                    'Grape___Esca_(Black_Measles)', 
                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 
                    'Orange___Haunglongbing_(Citrus_greening)', 
                    'Peach___Bacterial_spot',
                    'Peach___healthy', 
                    'Pepper,_bell___Bacterial_spot', 
                    'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 
                    'Potato___Late_blight', 
                    'Potato___healthy', 
                    'Raspberry___healthy', 
                    'Soybean___healthy', 
                    'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 
                    'Strawberry___healthy', 
                    'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 
                    'Tomato___Late_blight', 
                    'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 
                    'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
      ]
    st.success("Model is predicting it's a {}".format(class_names[result_index]))