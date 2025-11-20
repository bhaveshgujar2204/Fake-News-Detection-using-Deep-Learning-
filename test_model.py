from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf

# Load model
model_path = "./bertv3_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = TFBertForSequenceClassification.from_pretrained(model_path)

# Test input
text = "Breaking: Earth is Planet"

# Preprocess
inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)

# Predict
logits = model(inputs['input_ids'])[0]
prediction = tf.argmax(logits, axis=1).numpy()[0]

# Print result
label = "FAKE" if prediction == 1 else "REAL"
print(f"Prediction: {label}")
