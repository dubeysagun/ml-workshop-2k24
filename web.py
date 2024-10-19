from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.corpus import stopwords
import pandas as pd 

# Load the pre-trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text_data):
    preprocessed_text = []

    for sentence in text_data:
        if isinstance(sentence, str):  # Ensure the item is a string
            # Remove punctuation
            sentence = re.sub(r'[^\w\s]', '', sentence)
            # Tokenize, convert to lowercase, and remove stopwords
            preprocessed_sentence = ' '.join(token.lower()
                                             for token in sentence.split(' ')
                                             if token.lower() not in stopwords.words('english'))
            preprocessed_text.append(preprocessed_sentence)
        else:
            # Handle non-string items
            preprocessed_text.append('')

    return preprocessed_text

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the news text from the request
    news_text = request.form['news']
    # preprocess the news text
    news_text = pd.Series(preprocess_text([news_text]))
    news_text = vectorizer.transform(news_text)
    prediction = model.predict(news_text)
    if(prediction[0]==1):
        result = 'Real'
    else:
        result = 'Fake'

    # Render the template with the prediction result and class
    return render_template('index.html', prediction_text=f'The news is {result}.')

if __name__ == '__main__':
    app.run(debug=True)