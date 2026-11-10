
import streamlit as st
import numpy as np
import pickle
import json
import random
import re
import tensorflow as tf

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AB_AI",
    page_icon="🤖",
    layout="wide"
)

# ---------------- LOAD DEEP LEARNING MODEL ----------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("chatbot_model.keras")

    with open("words.pkl", "rb") as f:
        words = pickle.load(f)

    with open("classes.pkl", "rb") as f:
        classes = pickle.load(f)

    with open("intents.json", "r", encoding="utf-8") as f:
        intents = json.load(f)

    return model, words, classes, intents


# ---------------- CHATBOT RESPONSE ----------------
def predict_response(message, model, words, classes, intents):
    tokens = re.findall(r"\b\w+\b", message.lower())
    normalized = " ".join(tokens)

    # Reliable handling of common messages
    if normalized in {
        "hi", "hello", "hey", "greetings",
        "good morning", "good afternoon",
        "good evening", "how are you"
    }:
        target_tag = "greeting"

    elif normalized in {
        "bye", "goodbye", "see you later",
        "talk to you later", "take care"
    }:
        target_tag = "goodbye"

    elif normalized in {
        "thanks", "thank you", "thanks a lot",
        "i appreciate it", "thank you so much"
    }:
        target_tag = "thanks"

    else:
        # Deep-learning prediction for other messages
        bag = np.array(
            [1 if word in tokens else 0 for word in words],
            dtype=np.float32
        )

        # Avoid predictions when no known words are present
        if not np.any(bag):
            return (
                "I'm still learning that topic! 😊 "
                "Try saying hello, thanks, or goodbye."
            )

        prediction = model.predict(
            np.array([bag]), verbose=0
        )[0]

        index = int(np.argmax(prediction))
        confidence = float(prediction[index])

        if confidence < 0.70:
            return (
                "I'm still learning that topic! 😊 "
                "Try saying hello, thanks, or goodbye."
            )

        target_tag = classes[index]

    # Find a response for the selected intent
    for intent in intents["intents"]:
        if intent["tag"] == target_tag:
            return random.choice(intent["responses"])

    return "Sorry, I couldn't find a response."


model, words, classes, intents = load_model()


# ---------------- LIGHT JARVIS THEME ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fbff, #edf5ff);
    color: #183153;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1450px;
}

h1, h2, h3, h4, p, label {
    color: #183153 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #dceaff;
}

[data-testid="stSidebar"] * {
    color: #183153 !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #ffffff !important;
    border: 1px solid #dceaff;
    border-radius: 18px;
    padding: 12px;
    box-shadow: 0 5px 18px rgba(40, 100, 180, 0.06);
}

[data-testid="stChatMessage"] p {
    color: #183153 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background-color: #ffffff !important;
    border: 2px solid #a9ceff !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 15px rgba(40, 100, 180, 0.08);
}

[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #183153 !important;
    -webkit-text-fill-color: #183153 !important;
    caret-color: #1677ff !important;
    font-size: 16px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #7186a5 !important;
    -webkit-text-fill-color: #7186a5 !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #428bff !important;
    box-shadow: 0 0 0 3px rgba(66, 139, 255, 0.12);
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #e8f2ff, #d9eaff) !important;
    color: #174a8b !important;
    border: 1px solid #c5ddff !important;
    border-radius: 12px !important;
    padding: 10px 15px;
}

div.stButton > button:hover {
    background: #cde3ff !important;
    border-color: #78b2ff !important;
}

/* Information cards */
.jarvis-card {
    background: linear-gradient(145deg, #ffffff, #edf5ff);
    color: #183153;
    padding: 22px;
    border-radius: 22px;
    border: 1px solid #d9e8ff;
    box-shadow: 0 8px 28px rgba(35, 100, 180, 0.09);
    margin-bottom: 18px;
}

.robot {
    font-size: 90px;
    text-align: center;
    padding: 18px;
    background: radial-gradient(circle, #d9ebff, #f8fbff 70%);
    border-radius: 20px;
}

.small-text {
    color: #7186a5 !important;
    font-size: 14px;
}

.status {
    color: #209b64 !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# 🤖 Jarvis AI")
    st.caption("YOUR INTELLIGENT COMPANION")
    st.divider()

    st.markdown("### ✨ Navigation")
    st.markdown("💬 **Chat Assistant**")
    st.markdown("🧠 Deep Learning")
    st.markdown("⚙️ Project Information")

    st.divider()
    st.markdown("### 🟢 System Status")
    st.markdown("**● Model Loaded**")
    st.caption("TensorFlow • Neural Network")

    st.divider()
    st.caption("Dream big. Execute daily.")


# ---------------- HEADER ----------------
left, right = st.columns([3, 1])

with left:
    st.markdown("# Hello, I'm Jarvis! 👋")
    st.markdown(
        "Your personal chatbot, powered by **Deep Learning**."
    )

with right:
    st.markdown("""
    <div class="jarvis-card" style="text-align:center;">
        <span class="status">● ONLINE</span><br>
        <span class="small-text">AI ASSISTANT</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ---------------- CHAT LAYOUT ----------------
chat_column, info_column = st.columns([2.2, 1])

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! 👋 I'm Jarvis. How can I help you today?"
    }]

with chat_column:
    st.markdown("### 💬 Conversation")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Type your message to Jarvis...")

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        response = predict_response(
            user_input, model, words, classes, intents
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        st.rerun()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------------- RIGHT-SIDE INFORMATION ----------------
with info_column:
    st.markdown("""
    <div class="jarvis-card">
        <div class="robot">🤖</div>
        <h2 style="text-align:center;">JARVIS</h2>
        <p style="text-align:center;color:#7186a5;">
            Deep Learning Chatbot
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚡ System Features")

    st.markdown("""
    <div class="jarvis-card">
        <h4>🧠 Neural Network</h4>
        <p class="small-text">
            Intent prediction using TensorFlow.
        </p>
    </div>

    <div class="jarvis-card">
        <h4>💬 Smart Responses</h4>
        <p class="small-text">
            Responses based on trained intents.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚀 Quick Messages")

    quick_messages = {
        "👋 Say Hello": "Hello",
        "💙 Say Thanks": "Thanks",
        "👋 Say Goodbye": "Goodbye"
    }

    for label, message_text in quick_messages.items():
        if st.button(label, use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": message_text
            })

            response = predict_response(
                message_text, model, words, classes, intents
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

            st.rerun()


# ---------------- FOOTER ----------------
st.divider()
st.caption("JARVIS AI • Python • Streamlit • TensorFlow")