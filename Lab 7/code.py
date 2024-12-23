import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.utils import to_categorical 
# Step 1: Load and preprocess the MNIST dataset 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
# Reshape data to add a channel dimension (28x28x1 for grayscale images) 
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0 
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0 
# One-hot encode the labels 
y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test, 10) 
# Step 2: Build the CNN model 
model = Sequential([ 
Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)), 
MaxPooling2D(pool_size=(2, 2)), 
Conv2D(64, kernel_size=(3, 3), activation='relu'), 
MaxPooling2D(pool_size=(2, 2)), 
Flatten(), 
Dense(128, activation='relu'), 
Dropout(0.5), 
Dense(10, activation='softmax') 
]) 
# Step 3: Compile the model 
model.compile(optimizer='adam', 
loss='categorical_crossentropy', 
metrics=['accuracy']) 
# Step 4: Train the model 
history = model.fit(x_train, y_train,  
epochs=10,  
batch_size=64,  
validation_data=(x_test, y_test)) 
# Step 5: Evaluate the model 
test_loss, test_accuracy = model.evaluate(x_test, y_test) 
print(f"Test Accuracy: {test_accuracy:.2f}") 
# Step 6: Predict on sample data 
predictions = model.predict(x_test[:5]) 
for i, prediction in enumerate(predictions): 
    print(f"Sample {i} - Predicted Label: {tf.argmax(prediction).numpy()}, True Label:{tf.argmax(y_test[i]).numpy()}")
        
