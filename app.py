import streamlit as st
import openai
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit app setup
st.title("GPT Chat Interface")

# Set OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_response(prompt):
    """Generates a response from OpenAI's GPT-3 model."""
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

# Input text from the user
user_input = st.text_input("Enter your message:", "Hello, how can I connect with my spiritual side?")

if user_input:
    # Process input with SpaCy
    doc = nlp(user_input)
    st.write("## Processed Input:")
    for token in doc:
        st.write(f"{token.text} ({token.pos_})")
    
    # Generate response
    response = generate_response(user_input)
    st.write("## GPT-3 Response:")
    st.write(response)

st.write("*This app utilizes OpenAI's GPT-3 and SpaCy for natural language processing.")
