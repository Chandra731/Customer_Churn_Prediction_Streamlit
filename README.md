
# Customer Churn Prediction with Streamlit

This repository contains a machine learning-based customer churn prediction model, developed using **TensorFlow** and **scikit-learn**, and deployed with a **Streamlit** web application. The application allows users to input customer details and predict whether a customer is likely to churn, i.e., leave the service.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [How It Works](#how-it-works)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Model Training](#model-training)
7. [License](#license)

---

## Project Overview
Customer churn is a critical metric for any business, especially those with recurring revenue models. This project aims to predict the likelihood of a customer churning based on specific attributes, such as their credit score, geography, and account balance.

The prediction model was built using TensorFlow, trained on a customer churn dataset, and deployed with Streamlit to create a user-friendly interface where users can input customer data and receive a churn probability score.

## Technologies Used
- **Python**: Core programming language used.
- **TensorFlow**: Deep learning framework used to build the prediction model.
- **scikit-learn**: Used for data preprocessing and model evaluation.
- **Pandas**: Data manipulation and analysis.
- **Streamlit**: Web framework to build the interactive app.
- **Pickle**: For saving and loading the model and encoders.
  
## How It Works
The application takes user input in the form of customer details, including:
- Credit score
- Geography (France, Germany, Spain)
- Gender
- Age
- Tenure
- Balance
- Number of products
- Whether the customer has a credit card
- Whether the customer is an active member
- Estimated salary

The input data is processed, categorical features are encoded, and the features are scaled using a pre-trained model to predict the likelihood of churn. If the churn probability is higher than 0.5, the customer is predicted to churn.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   `git clone https://github.com/Chandra731/Customer_Churn_Prediction_Streamlit.git
   cd customer-churn-prediction `

2. Install the necessary dependencies:
   `pip install -r requirements.txt `

3. Run the Streamlit app:
   `streamlit run app.py `

The app will run locally at `http://localhost:8501`.

## Usage
Once the app is running:
1. Enter the customer's details in the provided fields.
2. Click on the **Predict Churn** button.
3. The app will display the predicted churn probability, with either a success or error message based on the prediction.

## Model Training
The model used for this app was trained using the **Churn_Modelling.csv** dataset. The dataset includes details such as credit score, geography, age, tenure, and other important features. The preprocessing involved:
- Encoding categorical features (`Geography`, `Gender`).
- Scaling the numerical features using `StandardScaler`.

The model architecture consists of a simple neural network with the following layers:
- Input layer with 64 units and ReLU activation.
- Hidden layer with 32 units and ReLU activation.
- Output layer with 1 unit and sigmoid activation for binary classification (churn vs. no churn).

To retrain the model:
1. Use the code provided in `model_training.ipynb`.
2. After training, save the model using TensorFlow’s `model.save()` method.

## License
This project is licensed under the MIT License. Feel free to use and modify the code for your own purposes.
