import streamlit as st
import pickle 
import sklearn 


# Load the trained sentiment analysis model

loaded_LR_model=pickle.load(open(r"G:\python applications\text_proj\models\lr_model.sav","rb")) 
loaded_vectorizer=pickle.load(open(r"G:\python applications\text_proj\models\count_vectorizer.sav","rb"))  

# Define a function to predict the sentiment
def predict(text):
    
    # Vectorize using bow vectorizer
    text_vector = loaded_vectorizer.transform([text])

    # Make predictions using the loaded model
    sentiment_prediction = loaded_LR_model.predict(text_vector)

    return sentiment_prediction[0]   

# Create the Streamlit web app
def main():
    st.title(" Text Sentiment Analysis Web App")
    st.write("🚀 Welcome to the Text Sentiment Explorer! Uncover the Emotions in Text. 📊")

    # Get user input
    user_input = st.text_area("Drop your text here and let's reveal its sentiment:")

    if st.button("Predict Sentiment 🔍"):
    
        if user_input.strip() != '':
            sentiment = predict(user_input)
            st.write(f'🔍 Sentiment Analysis Result: {sentiment}')

        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()