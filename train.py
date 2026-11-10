import json
import pickle
import random
import re
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

# Load our dataset
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

words = []
classes = []
documents = []

# Prepare the training data
for intent in data["intents"]:
    for pattern in intent["patterns"]:

        tokens = re.findall(r"\b\w+\b", pattern.lower())

        words.extend(tokens)
        documents.append((tokens, intent["tag"]))

    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = sorted(set(words))
classes = sorted(set(classes))

# Save words and intent labels
with open("words.pkl", "wb") as file:
    pickle.dump(words, file)

with open("classes.pkl", "wb") as file:
    pickle.dump(classes, file)

# Convert messages into numerical features
training_data = []

for tokens, tag in documents:

    bag = [1 if word in tokens else 0 for word in words]

    output = [0] * len(classes)
    output[classes.index(tag)] = 1

    training_data.append([bag, output])

random.shuffle(training_data)

X = np.array([item[0] for item in training_data])
y = np.array([item[1] for item in training_data])

# Build our Deep Learning neural network
model = Sequential([
    Input(shape=(len(words),)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(len(classes), activation="softmax")
])

# Configure the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(X, y, epochs=200, verbose=1)

# Save the trained model
model.save("chatbot_model.keras")

print("\nTraining completed successfully!")
print("Vocabulary:", words)
print("Recognized intents:", classes)
print("Model saved as chatbot_model.keras")