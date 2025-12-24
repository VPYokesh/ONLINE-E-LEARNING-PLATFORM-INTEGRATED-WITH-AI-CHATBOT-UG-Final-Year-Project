import openai
import os
import base64
from flask import Flask, render_template, request, jsonify
from chat import get_response


app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.php")

@app.post("/predict")
def predict():

    text = request.get_json().get("message")
    # Check if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)


openai.api_key = os.environ.get("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")

@app.route('/')
def index():
    return render_template('base.php')

@app.route('/generate_response', methods=['POST'])
def generate_response():
    data = request.get_json()
    text = data.get('text')

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text}
        ],
        api_key=openai.api_key
    )

    ai_text = response.choices[0].message.content
    return jsonify({'ai_text': ai_text})


if __name__ == "__main__":
    app.run(debug=True)

