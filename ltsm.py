import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Loading preprocessed tweets and encoded sentiments from previous step
processed_tweets = [...]
encoded_sentiments = [...]

# Creating tokenizer to convert the tweets to sequences of integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(processed_tweets)
sequences = tokenizer.texts_to_sequences(processed_tweets)

max_length = max([len(s) for s in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Spliting the dataset
train_size = int(len(padded_sequences) * 0.8)
X_train, X_test = padded_sequences[:train_size], padded_sequences[train_size:]
y_train, y_test = encoded_sentiments[:train_size], encoded_sentiments[train_size:]

# LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_length))
model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

