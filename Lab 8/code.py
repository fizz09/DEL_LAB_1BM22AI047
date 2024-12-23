# Import necessary libraries 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
# Step 1: Load and preprocess the dataset 
# Load the IMDB dataset with top 10,000 most common words 
vocab_size = 10000 
max_length = 200 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size) 
# Pad sequences to ensure uniform input size 
x_train = pad_sequences(x_train, maxlen=max_length) 
x_test = pad_sequences(x_test, maxlen=max_length) 
# Step 2: Build the RNN model 
model = Sequential([ 
Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length), 
SimpleRNN(128, activation='tanh', return_sequences=False), 
Dropout(0.5), 
Dense(64, activation='relu'), 
Dropout(0.5), 
Dense(1, activation='sigmoid') 
]) 
# Step 3: Compile the model 
model.compile(optimizer='adam', 
loss='binary_crossentropy', 
metrics=['accuracy']) 
# Step 4: Train the model 
history = model.fit(x_train, y_train,  
epochs=5,  
batch_size=64,  
validation_data=(x_test, y_test)) 
# Step 5: Evaluate the model 
test_loss, test_accuracy = model.evaluate(x_test, y_test) 
print(f"Test Accuracy: {test_accuracy:.2f}") 
# Step 6: Predict sentiment for a sample review 
sample_review = "This movie was fantastic! The characters were well-developed and the plot was thrilling." 
# Tokenize and pad the sample review 
from tensorflow.keras.preprocessing.text import Tokenizer 
tokenizer = imdb.get_word_index() 
reverse_word_index = {value: key for (key, value) in tokenizer.items()} 
encoded_review = [tokenizer.get(word, 2) for word in sample_review.lower().split()] 
padded_review = pad_sequences([encoded_review], maxlen=max_length) 
# Predict sentiment 
prediction = model.predict(padded_review) 
sentiment = "Positive" if prediction[0] > 0.5 else "Negative" 
print(f"Predicted Sentiment: {sentiment}")
