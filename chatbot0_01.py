pip install transformers

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

# Define a conversation history list to store the chat history
conversation_history = []

# Set page title
st.title("ChatBot")

# Create chatbox widget
user_input = st.text_input("User")

# Check if user has entered any input
if user_input:
    # Append the user input to the conversation history
    conversation_history.append(user_input)

    # Tokenize and encode the conversation history
    encoded_inputs = tokenizer([user_input] + conversation_history[:-1], padding=True, truncation=True, return_tensors="pt")

    # Generate a response
    response = model.generate(encoded_inputs.input_ids, max_length=100)

    # Decode the response
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)

    # Append the response to the conversation history
    conversation_history.append(response_text)

    # Display the response
    st.text_area("ChatBot", value=response_text, height=100)

# Add a button to clear the conversation history
if st.button("Clear Conversation"):
    conversation_history = []

