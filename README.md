# 🤖 Jarvis AI Chatbot

### A Deep Learning Based Conversational Chatbot

## 📌 About the Project

Jarvis AI Chatbot is a simple conversational chatbot developed using Python and Deep Learning. It uses an Artificial Neural Network (ANN) to understand user messages and generate suitable responses based on a predefined intents dataset.

The chatbot is designed to provide basic conversations through a user-friendly interface.

## 🎯 Project Objectives

- To understand the basics of Deep Learning.
- To implement an Artificial Neural Network (ANN).
- To train a chatbot using a custom intents dataset.
- To generate responses based on user input.
- To build a simple and interactive chatbot interface.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep Learning framework |
| Keras | Building and training the ANN model |
| ANN | Chatbot prediction model |
| JSON | Storing chatbot intents and responses |
| Streamlit | User interface |
| Flask | Web application backend |

## 🧠 Deep Learning Implementation

This project uses an Artificial Neural Network (ANN) for chatbot training.

The working process:

1. **Dataset:** User messages and responses are stored in `intents.json`.
2. **Preprocessing:** Input text is converted into a numerical representation.
3. **Training:** The ANN model learns patterns from the training data.
4. **Prediction:** The trained model predicts the appropriate intent.
5. **Response:** The chatbot provides a response based on the predicted intent.

## 📂 Project Structure

```text
DL_Chatbot/
│
├── app.py
├── streamlit_app.py
├── train.py
├── intents.json
├── chatbot_model.keras
├── words.pkl
├── classes.pkl
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project

```bash
cd DL_Chatbot
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install tensorflow==2.20.0 streamlit flask
```

Install any additional packages required by the application.

## ▶️ Run the Project

To run the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

To run the Flask application:

```bash
python app.py
```

## 📚 Learning Outcomes

- Understanding Artificial Neural Networks.
- Practical implementation of TensorFlow and Keras.
- Basic Natural Language Processing concepts.
- Training and using a Deep Learning model.
- Building a chatbot application using Python.

## 👨‍💻 Developer

**Kanuru Sai Aditya**

B.Tech – Computer Science and Engineering

## 📌 Project Type

Academic Mini Project – Deep Learning

---

⭐ If you find this project interesting, feel free to explore the repository!
