# Lung Cancer Detection with X-Ray/CT Images

This project aims to detect different types of lung cancer from X-ray or CT images using deep learning. The model is based on a Convolutional Neural Network (CNN) and predicts cancer types from uploaded lung X-ray or CT scan images. The project is implemented using **Streamlit** for the web interface and **TensorFlow/Keras** for the model.

## Key Features

- **Image Upload**: Allows users to upload X-ray or CT images of the lungs.
- **Prediction**: The model predicts the type of cancer with a confidence score.
- **Multi-language Support**: The web interface supports multiple languages (English, Hindi, Punjabi).
- **Image Augmentation**: The app provides an option to apply image augmentation for better model generalization.
- **Model Architecture**: Option to view the model architecture.

## Classes Predicted

The model can predict the following classes:
- **Adenocarcinoma**
- **Large Cell Carcinoma**
- **Normal** (No signs of cancer)
- **Squamous Cell Carcinoma**

## Setup

To run this project locally, follow the steps below:

### Prerequisites
Ensure that you have the following installed on your machine:
- Python 3.x
- TensorFlow
- Keras
- Streamlit
- Other dependencies (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/marslanm1/Lung-Cancer-Detection.git
   cd Lung-Cancer-Detection

	2.	Install the required packages:

pip install -r requirements.txt



Running the Application

To run the Streamlit app, use the following command:

streamlit run app.py

This will start the application, and you can open it in your browser.

Model Details

The deep learning model used for prediction is a Convolutional Neural Network (CNN), trained on a dataset of lung X-ray/CT images. The model classifies the images into four categories:
	1.	Adenocarcinoma
	2.	Large Cell Carcinoma
	3.	Normal
	4.	Squamous Cell Carcinoma

The model provides a confidence score for its predictions, which indicates the model’s certainty.

Performance Metrics
	•	Accuracy: The model’s accuracy on the test set.
	•	Precision: The precision of the model for each class.
	•	Recall: The recall of the model for each class.
	•	F1-Score: The F1-score for each class.

Example Usage
	1.	Upload an image: Choose an X-ray or CT image of the lungs.
	2.	Prediction: Click the “Predict” button to get the cancer type prediction with confidence score.
	3.	Save the results: You can download the prediction results for further analysis.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
	•	Thanks to TensorFlow and Keras for providing excellent deep learning frameworks.
	•	Special thanks to the contributors of open-source projects that made this project possible.
