from flask import Flask, render_template, request
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import re
from nltk.stem import PorterStemmer

# Initialize the Flask app
app = Flask(__name__)

# Load the trained BERT model
model = TFBertForSequenceClassification.from_pretrained('/content/bertv3_model')  # Adjust path if necessary
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocessing function (same as before)
def preprocess_text(text):
    stemmer = PorterStemmer()
    # Replace non-alphabet characters with spaces
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = [stemmer.stem(word) for word in text.split()]
    text = ' '.join(text)
    return text

# Home route: Render the index.html
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route: Render prediction.html and handle prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = ''
    if request.method == 'POST':
        input_text = request.form['news']
        
        # Preprocess the input text
        preprocessed_text = preprocess_text(input_text)
        
        # Tokenize and prepare inputs for BERT
        inputs = tokenizer(preprocessed_text, truncation=True, padding='max_length', max_length=42, return_tensors='tf')
        token_tensors = inputs['input_ids']
        segment_tensors = inputs['token_type_ids']
        mask_tensors = inputs['attention_mask']

        # Make prediction
        predictions = model.predict([token_tensors, segment_tensors, mask_tensors])
        logits = predictions.logits[0]
        probabilities = tf.nn.softmax(logits)
        predicted_label = tf.argmax(probabilities)
        
        if predicted_label == 0:
            prediction_text = f"*-*-Fake News-*-*\nProbability of being fake: {probabilities[0]*100:.2f}%\nProbability of being real: {probabilities[1]*100:.2f}%"
        else:
            prediction_text = f"*-*-Real News-*-*\nProbability of being fake: {probabilities[0]*100:.2f}%\nProbability of being real: {probabilities[1]*100:.2f}%"
    
    return render_template('prediction.html', prediction_text=prediction_text)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
