from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import json
import random
import re

app = Flask(__name__)

# Load the trained Deep Learning model
model = tf.keras.models.load_model("chatbot_model.keras")

# Load vocabulary and intent classes
with open("words.pkl", "rb") as file:
    words = pickle.load(file)

with open("classes.pkl", "rb") as file:
    classes = pickle.load(file)

with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)


def predict_intent(message):
    tokens = re.findall(r"\b\w+\b", message.lower())

    # Convert the message into a numerical bag of words
    bag = [1 if word in tokens else 0 for word in words]
    prediction = model.predict(
        np.array([bag]), verbose=0
    )[0]

    intent_index = int(np.argmax(prediction))
    confidence = float(prediction[intent_index])

    # Avoid confident guesses for completely unknown messages
    if confidence < 0.60:
        return "unknown"

    return classes[intent_index]


def get_response(message):
    intent_tag = predict_intent(message)

    if intent_tag == "unknown":
        return "Sorry, I don't understand that yet. Try saying hello!"

    for intent in intents["intents"]:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])

    return "Sorry, something went wrong."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"response": "Please type a message."})

    response = get_response(message)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)

