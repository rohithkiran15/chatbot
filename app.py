from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util
import csv

app = Flask(__name__)

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')


# Load data from CSV
def load_data(filename):
    data = {}
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data[row["Question"].strip()] = row["Answer"].strip()
    return data


filename = "cleaned_data4.csv"  # Replace with your CSV file
data = load_data(filename)


# Chatbot logic
def get_chatbot_response(user_input, top_k=3):
    questions = list(data.keys())
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, question_embeddings)
    top_results = similarities[0].topk(k=top_k)

    # Collect answers based on similarity scores above a threshold (e.g., 0.5)
    answers = [
        data[questions[idx]]
        for idx, score in zip(top_results.indices.tolist(), top_results.values.tolist())
        if score > 0.5
    ]

    if not answers:
        return ["I'm sorry, I don't understand that."]

    return answers  # Return a list of answers, not just the first one


# Routes
@app.route('/')
def home():
    return render_template('chat_direct.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('message')
    response = get_chatbot_response(user_input)
    return render_template('chat_direct.html', user_input=user_input, chatbot_response=response)


if __name__ == '__main__':
    app.run(debug=True)
