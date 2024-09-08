import pandas as pd
import re
import numpy as np
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle

# Functions for cleaning data
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

def remove_special_characters(text):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

def simple_stemmer(text):
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def remove_stopwords(text):
    stopwords_set = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords_set]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def preprocessing(df):
    df['review'] = df['review'].apply(denoise_text)
    df['review'] = df['review'].apply(remove_special_characters)
    df['review'] = df['review'].apply(simple_stemmer)
    df['review'] = df['review'].apply(remove_stopwords)
    return df

def create_lstm_model(max_words, max_len):
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_len))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    return model

def create_lstm_conv1d_model(max_words, max_len):
    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_len))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(2, activation='softmax'))
    return model

def plot_history(history):
    # Plot loss
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('lstm_conv1d_model_loss_plot.png')
    
    # Plot accuracy
    plt.figure(figsize=(10,5))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('lstm_conv1d_model_accuracy_plot.png')

def main():
    # Load data
    imdb_data=pd.read_csv('./data-movie/movie_data.csv')

    # Preprocess reviews
    imdb_data = preprocessing(imdb_data)

    max_words = 5000
    max_len = 250
    tokenizer = Tokenizer(num_words=max_words, split=' ')
    tokenizer.fit_on_texts(imdb_data['review'].values)

    # Save the tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X = tokenizer.texts_to_sequences(imdb_data['review'].values)
    X = pad_sequences(X, maxlen=max_len)
    
    # Convert sentiments to binary
    le = LabelEncoder()
    Y = le.fit_transform(imdb_data['sentiment'])
    
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    
    # Create the model
    model = create_lstm_conv1d_model(max_words, max_len)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Fit the model
    history = model.fit(X_train, Y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1)
    model.save('lstm_conv1d_model.hdf5')
    
    # Plot the training history
    plot_history(history)
    
    # Evaluate the model
    score, acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = 64)
    print("The test accuracy of the model is: ", acc)

main()
