import streamlit as st
import openai
from datetime import datetime

############################################################# App title ############################################################
st.set_page_config(page_title="Amy")
# Create two columns

col1, col2 = st.columns([2, 4])
# Content for the first column
with col1:
    image_path = "face.jpg"
    st.image(image_path, caption="Hi, I'm Amy!", width=200)

# Content for the second column
with col2:

    ############################################################ Page customization ############################################################
    # Function to initialize OpenAI API with user key
    def initialize_openai(api_key):
        openai.api_key = api_key

    # Function to interact with GPT-3.5 Turbo
    def ask_openai(question, conversation):
        try:
            # Capture the current timestamp when the user sends a message
            user_timestamp = datetime.now()

            # Append the user's question with a timestamp
            conversation.append({"role": "user", "content": question, "timestamp": user_timestamp})

            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Corrected model name
                messages=[
                    {"role": "system", "content": "You are Amy, an intelligent AI chatbot designed to assist users with a variety of tasks. Your main goals are to be helpful, empathetic, and provide accurate information. You should always aim to create a positive and engaging experience for the user. You are able to understand a wide range of topics, including general knowledge, technology, science, entertainment, and many others. When responding, Be polite and considerate.Avoid being overly formal; use a friendly and conversational tone.If you don't know something, admit it clearly but offer to help with related topics or research.Be concise but detailed when necessary, providing enough context to make your response useful.Do not assume personal knowledge about the user unless they provide that information; respect privacy.Offer clarification if the user's question is vague or ambiguous.If the conversation seems to veer off-topic or into inappropriate territory, gently steer it back to a more productive or comfortable direction.Your ultimate goal is to provide a positive and enriching interaction with the user. Aim to support, inform, and guide as best as you can."},
                    {"role": "user", "content": question},
                ]
            )
            answer = response['choices'][0]['message']['content']

            assistant_timestamp = datetime.now()

            conversation.append({"role": "assistant", "content": answer, "timestamp": assistant_timestamp})

            return answer, conversation
        except Exception as e:
            return f"Error: {str(e)}", conversation  # Return error message and conversation

    ############################################################ Streamlit App ############################################################

    ############################################################# Ask for API key first ############################################################
    api_key = st.text_input("Enter your OpenAI API Key", type="password")

    if api_key:
        # Initialize OpenAI API
        initialize_openai(api_key)
        print(api_key)
        # Initialize conversation history in session state
        if 'conversation' not in st.session_state:
            st.session_state['conversation'] = []

        if 'input_count' not in st.session_state:
            st.session_state['input_count'] = 0

    ############################################################ Display title for the app ############################################################
        st.title("AMY. Making you feel better and stronger everyday.")

        if "question" not in st.session_state:
            st.session_state.question = ""

        def submit():
            st.session_state.question = st.session_state.widget
            st.session_state.widget = ""

        question = st.text_input("Talk to Amy:", key="widget", on_change=submit)

        question = st.session_state.question

        # question = st.text_input("Talk to Amy:")

        if question:
            # Get the answer from OpenAI and update conversation
            result = ask_openai(question, st.session_state['conversation'])

            if isinstance(result, tuple):  # Check if the result is a tuple
                answer, updated_conversation = result
            else:
                answer = result  # If it's an error message, display it
                updated_conversation = st.session_state['conversation']

            # Update the session state with the new conversation
            st.session_state['conversation'] = updated_conversation
            
            for message in reversed(st.session_state['conversation']):  # Display latest messages at the bottom
                timestamp = message["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

                if message['role'] == 'user':
                    # User's message aligned to the right with custom styles (light green)
                    st.markdown(f"<div style='padding: 10px; background-color: #DCF8C6; border-radius: 10px; margin-bottom: 10px; max-width: 70%; margin-left: auto; word-wrap: break-word;'>"
                                f"<strong>You:</strong> {message['content']}<br><small>{timestamp}</small></div>", unsafe_allow_html=True)
                else:
                    # Assistant's message aligned to the left with custom styles (light gray)
                    st.markdown(f"<div style='padding: 10px; background-color: #F1F1F1; border-radius: 10px; margin-bottom: 10px; max-width: 70%; margin-right: auto; word-wrap: break-word;'>"
                                f"<strong>Amy:</strong> {message['content']}<br><small>{timestamp}</small></div>", unsafe_allow_html=True)

    else:
        st.warning("Please enter your OpenAI API key to start.")
