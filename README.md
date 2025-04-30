🩺 Lung Cancer Detection from X-ray and CT Images

This project involves building a deep learning model to predict lung cancer types (Adenocarcinoma, Squamous Cell Carcinoma, Large Cell Carcinoma, and Normal) based on X-ray and CT scan images. The model is built using TensorFlow and Keras, and it is deployed using Streamlit for interactive predictions.

Table of Contents
	1.	Overview
	2.	Features
	3.	Model Details
	4.	How to Use
	5.	Dataset
	6.	License
	7.	Acknowledgements

Overview

This web application allows users to upload lung X-ray or CT scan images and receive predictions on the type of lung cancer or whether the image is normal. The project utilizes a pre-trained convolutional neural network (CNN) model that is deployed via a user-friendly Streamlit interface.

Features
	•	Image Upload: Upload lung X-ray or CT scan images (JPEG, PNG, or JPG).
	•	Prediction: Get predictions with confidence scores.
	•	Cancer Details: Provides descriptions of the detected cancer type in multiple languages (English, Hindi, Punjabi).
	•	Save Results: Option to download the prediction results in CSV format.
	•	Image Augmentation: Apply augmentation techniques like rotation, width/height shift, and zoom for enhancing the dataset.
	•	Model Architecture: View the model architecture and understand its layers.

Model Details

The model is a CNN-based deep learning model trained to classify lung cancer into one of the following categories:
	•	Adenocarcinoma
	•	Large Cell Carcinoma
	•	Squamous Cell Carcinoma
	•	Normal

Model Specifications:
	•	Framework: TensorFlow 2.x
	•	Preprocessing: Image resizing to (350, 350), normalization
	•	Architecture: Deep Convolutional Neural Network
	•	Final Model: final_model.h5

How to Use

Prerequisites
	•	Python 3.x
	•	TensorFlow
	•	Streamlit
	•	PIL (Python Imaging Library)
	•	Pandas
	•	Numpy
	•	Other necessary libraries

Installation
	1.	Clone the repository:

git clone https://github.com/marslanm1/Lung-Cancer-Detection.git
cd Lung-Cancer-Detection


	2.	Install the required dependencies:

pip install -r requirements.txt


	3.	Run the Streamlit app:

streamlit run app.py


	4.	Open the app in your browser at http://localhost:8501.

Upload Image
	•	Upload an X-ray or CT scan image of the lungs.
	•	The model will predict the cancer type with a confidence score and display the results.

## Dataset

The dataset used for training and testing the model is sourced from Kaggle:

- **Dataset**: [Chest CT Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
- **Creator**: Mohamed Hanyyy
- **License**: Please refer to the dataset's license on Kaggle for more information.
- 
Ensure you respect the license terms provided by Kaggle when using this dataset.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
	•	The model is based on convolutional neural networks (CNN) for image classification.
	•	Special thanks to Kaggle and the dataset creator Mohamed Hanyyy for providing the dataset used in this project.
