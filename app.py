import streamlit as st
import spacy
from transformers import pipeline

# Load Spacy NLP model for text processing
nlp = spacy.load('en_core_web_sm')

# Load GPT-based model for generating responses
chat_model = pipeline('text-generation', model='gpt2')

def generate_response(user_input):
    # Process the user input with Spacy
    doc = nlp(user_input)

    # Generate a response using the GPT-based model
    response = chat_model(user_input, max_length=150, num_return_sequences=1)
    return response[0]['generated_text']

# Streamlit UI setup
st.title("GPT Chat Interface")
st.write("Interact with an AI-powered chat interface!")

# User input box
user_input = st.text_input("You:", "")

# Display response if input is provided
if user_input:
    response = generate_response(user_input)
    st.text_area("AI:", response)
