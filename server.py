from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load the trained BERT model and tokenizer
MODEL_PATH = "./bertv3_model"

try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None  # Avoid using an uninitialized model

# Serve the HTML file
@app.route('/')
def index():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid JSON format"}), 400

        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Tokenize input text
        inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors="tf")

        # Get model prediction
        logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
        prediction_probs = tf.nn.softmax(logits).numpy()

        # Debugging: Print logits and probabilities
        print(f"Logits: {logits.numpy()}")
        print(f"Softmax Probabilities: {prediction_probs}")

        # Get predicted class
        predicted_label = np.argmax(prediction_probs, axis=1)[0]
        label_map = {0: "FAKE", 1: "REAL"}
        predicted_text = label_map.get(predicted_label, "UNKNOWN")

        return jsonify({"prediction": predicted_text, "probabilities": prediction_probs.tolist()})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
