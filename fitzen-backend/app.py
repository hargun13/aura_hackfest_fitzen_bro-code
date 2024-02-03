from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

nltk.download('popular')

app = Flask(__name__)
app.static_folder = 'static'
CORS(app)

###################### CHATBOT ##############################
intents = json.loads(open('C:\\Users\\HARGUN\\Desktop\\Aura Hackfest - fitzen\\fitzen-backend\\chatbot\\intents.json').read())
words = pickle.load(open('C:\\Users\\HARGUN\\Desktop\\Aura Hackfest - fitzen\\fitzen-backend\\chatbot\\texts.pkl', 'rb'))
classes = pickle.load(open('C:\\Users\\HARGUN\\Desktop\\Aura Hackfest - fitzen\\fitzen-backend\\chatbot\\labels.pkl', 'rb'))
model = load_model('C:\\Users\\HARGUN\\Desktop\\Aura Hackfest - fitzen\\fitzen-backend\\chatbot\\model.h5')
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    if ints:
        res = getResponse(ints, intents)
    else:
        res = "Sorry, I cannot answer this question for you."
    return res

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_text = request.get_json().get('msg')
    print(user_text)
    res = chatbot_response(user_text)
    # print(res)
    return jsonify({'response': res})
###################### CHATBOT ##############################



###################### MEAL PLAN ############################
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
HEADERS = {"Authorization": "Bearer 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"}

def query(payload):
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as err:
        print(f"Request error occurred: {err}")
        return None

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json(force=True)
        user_input = data.get('prompt')
        max_length = 200
        generated_text = ""

        while len(generated_text.split()) < max_length:
            response = query({
                "inputs": user_input,
                "max_length": 50
            })

            if response:
                generated_text += response[0]["generated_text"]
                user_input = generated_text.split()[-10:]
            else:
                break

        return jsonify({'generated_text': generated_text})
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': 'Invalid request format'}), 400
###################### MEAL PLAN ############################
    

if __name__ == "__main__":
    app.run(port=5000, debug=True)
