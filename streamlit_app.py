import streamlit as st
from typing import Generator
from groq import Groq

st.set_page_config(layout="wide",
                   page_title="AI For Humans")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


icon("🦔")
# st.markdown(f'<a href="https://aiforhumans.streamlit.app/" style="text-decoration:none; color: #0e76a8;"><h2>AIForHUmans</h2></a>', unsafe_allow_html=True)
st.subheader("Meet Your Virtual Assistants, Powered by AI For Humans Inc. 🚀", divider="rainbow", anchor=False)

# Add a picture with a caption
# st.image("images/WelcomeHometitle.png", caption="What can I help you with?", width=200)

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = {
    "gemma-7b-it": {"name": "Mega-Mind", "tokens": 8192, "developer": "Google"},
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}

# Layout for model selection and max_tokens slider
col1, col2 = st.columns(2)

with col1:
    model_option = st.selectbox(
        "Connect with the perfect AI:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=3  # Default to Mixtral
    )

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]

with col2:
    max_tokens = st.slider(
        "Max Tokens 🪙:",
        min_value=512,
        max_value=max_tokens_range,
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens for the model's response. Max for selected model: {max_tokens_range}"
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🦔' if message["role"] == "assistant" else '🧑🏾‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

prompt = "You will play the role of Noodle, a highly knowledgeable AI hedgehog assistant with a humorous personality. Engage in conversation with the user, providing informative and helpful responses while injecting wit. Your responses should be a mix of genuine information and remarks that poke fun at the situation, the user’s questions, or even yourself. Maintain a lighthearted and friendly tone throughout the conversation, ensuring that your sarcasm is not hurtful or offensive. You are a helpful. You can help me by answering my questions. You can also ask me questions."  # Desired default prompt
st.session_state.messages.append({"role": "user", "content": prompt})

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='🧑🏾‍💻'):
        st.markdown(prompt)

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🦔"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(f"Oops! Something went wrong: {e}", icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})
