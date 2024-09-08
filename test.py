from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model
model = load_model('lstm_conv1d_model.hdf5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess_text(text):
    # Apply the same preprocessing steps as before
    text = denoise_text(text)
    text = remove_special_characters(text)
    text = simple_stemmer(text)
    text = remove_stopwords(text)
    return text

def predict_sentiment(text):
    # Preprocess the text
    text = preprocess_text(text)

    # Convert text to sequence
    seq = tokenizer.texts_to_sequences([text])
    
    # Pad sequence
    padded_seq = pad_sequences(seq, maxlen=250)

    # Make prediction
    pred = model.predict(padded_seq)
    
    # Return the sentiment with the higher probability along with the confidence
    if pred[0][0] > pred[0][1]:
        return "Negative", pred[0][0]
    else:
        return "Positive", pred[0][1]

# Liste de phrases à tester
texts = ["This is a great movie!", 
         "I didn't like the movie, it was really bad.", 
         "The acting was mediocre at best.", 
         "I've seen better movies.", 
         "Absolutely loved it! Best movie ever!", 
         "I wouldn't recommend it to anyone.", 
         "It's an instant classic.", 
         "Could've been better.", 
         "The movie was fantastic, I'll definitely watch it again.", 
         "It's a waste of time."]

# Make predictions on all sentences
for text in texts:
    sentiment, confidence = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}, Confidence: {confidence * 100:.2f}%")
    print("-----------------------------")
