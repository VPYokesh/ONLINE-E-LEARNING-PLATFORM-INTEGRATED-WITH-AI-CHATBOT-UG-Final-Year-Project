import random
import json
import torch
import openai
import requests
from model import NeuralNet
from nltk_utils import tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "E_Learn"


def get_response(msg):
    words = tokenize(msg)
    response = "Hi, I'm E-Learn. How can I assist you?"

    # Check for exact matches in patterns
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if msg.lower() == pattern.lower():
                response = random.choice(intent['responses'])
                return response

    # Check for partial matches in patterns
    for intent in intents["intents"]:
        if any(msg.lower() in pattern.lower() for pattern in intent["patterns"]):
            response = random.choice(intent['responses'])
            return response

    # Check for keyword matches in patterns
    for intent in intents["intents"]:
        for keyword in intent.get("keywords", []):
            if keyword.lower() in msg.lower():
                for pattern in intent["patterns"]:
                    if keyword.lower() in pattern.lower():
                        response = random.choice(intent['responses'])
                        return response

    # If no answer is found in intents
    response = None

    if not response:
        # Search with Google only if no answer is found in intents
        response = search_with_google(msg)

    if not response:
        # If still no answer found, provide a default response
        response = "Sorry, I couldn't find an answer to your question."

    return response


def search_with_google(query):
    params = {
        'key': "AIzaSyDhLUr3HWbiWBFXF9XnxXKjPe97c7LCXhs",
        'cx': "936ab47c6bf704913",
        'q': query
    }

    response = requests.get('https://www.googleapis.com/customsearch/v1', params=params)
    data = response.json()

    if 'items' in data:
        # Extract the first search result
        first_result = data['items'][0]
        title = first_result.get('title')
        link = first_result.get('link')
        snippet = first_result.get('snippet')

        # Truncate the response text to exactly 1500 words
        max_words = 1500
        snippet_words = snippet.split()[:max_words]
        snippet = ' '.join(snippet_words)

        return f" {title} - {snippet} "

    return None