# Text-Sentiment-Prediction

This project focuses on sentiment analysis of customer reviews using machine learning techniques. The goal is to determine whether the sentiment behind a given customer review is positive or negative. The project involves data analysis, text preprocessing, model training, and creating a Streamlit web application for real-time sentiment prediction.

## Introduction

Sentiment analysis is a powerful technique that helps us understand the emotions and opinions expressed in textual data. In this project, we address sentiment analysis on customer reviews using various machine learning models. The process involves understanding the provided reviews, preprocessing the text data, training multiple models, and creating a user-friendly web application for sentiment prediction.

## Data Overview

The dataset consists of **568,454** text files, each representing a customer review. These reviews are labeled as positive or negative sentiments based on customer feedback. The initial step involves analyzing the data to gain insights into the distribution of sentiments and identifying trends.

[Link to Dataset](https://drive.google.com/file/d/1Pp1UPgmh7A_V5B0VEKQvpffa5kq_CieL/view?usp=sharing)

## Data Preprocessing

- Exploratory Data Analysis (EDA) is performed to understand the distribution of sentiments, common words, and trends.
- Text cleaning is applied to remove special characters, punctuation, and irrelevant information.
- The cleaned text is processed using the Bag of Words and TF-IDF techniques to convert text into numerical features for model training.

## Model Training

Multiple machine learning models are trained to classify sentiments:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree
- Random Forest

Each model is trained using both Bag of Words and TF-IDF preprocessed data.

## Model Evaluation

The models are evaluated based on accuracy and latency:

- Accuracy measures how well the models classify sentiments on test data.
- Latency measures the time taken by each model to make predictions.

Model performance is compared, and insights are drawn from the evaluation results.

## Web Application

A Streamlit web application is developed to provide a user-friendly interface for sentiment prediction. Users can input their text, and the application will predict whether the sentiment is positive or negative. The application utilizes the best-performing model from the evaluation stage.

[Watch the Project Video](https://drive.google.com/file/d/1ZxvyD9rSmijYdgx9DCDyViHxK1GzmlVY/view?usp=sharing)

## Contributing

Contributions are welcome! If you find any issues or want to enhance the project, feel free to open a pull request.
