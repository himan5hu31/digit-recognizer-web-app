# digit-recognizer-web-app

### MNIST Digit Recognizer using CNN
This is a project for training a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. The model is built using TensorFlow and Keras.

### Dataset
The MNIST dataset consists of 70,000 images of handwritten digits from 0 to 9. The images are grayscale and have a resolution of 28x28 pixels.

The dataset is available in Keras and can be easily loaded using the following code:


```

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
### Model Architecture
The model consists of two convolutional layers with ReLU activation, followed by two max pooling layers, and two fully connected layers with dropout regularization. The output layer has 10 units with softmax activation, representing the probabilities of the input image being each of the 10 digits.

The architecture of the model is as follows:
```

model1=keras.Sequential([
    keras.layers.Conv2D(16,(3,3),padding="same",activation='relu',input_shape=(28,28,1),name='Conv2D_1'),
    keras.layers.MaxPooling2D(2,2,name='MaxPool_1'),
    
    keras.layers.Conv2D(32,(3,3),padding="same",activation='relu',input_shape=(28,28,1),name='Conv2D_2'),
    keras.layers.MaxPooling2D(2,2,name='MaxPool_2'),
    
    keras.layers.Conv2D(64,(3,3),padding="same",activation='relu',input_shape=(28,28,1),name='Conv2D_3'),
    keras.layers.MaxPooling2D(2,2,name='MaxPool_3'),
    
    keras.layers.Conv2D(128,(3,3),padding="same",activation='relu',input_shape=(28,28,1),name='Conv2D_4'),
    keras.layers.MaxPooling2D(2,2,name='MaxPool_4'),
    
    keras.layers.Flatten(name='FL'),
    keras.layers.Dense(512,activation='relu',name='FC_layer'),
    keras.layers.Dense(10,activation='softmax')
    
])

model1._name = "HimBA"

model1.summary()
```
### Training
The model is trained using the Adam optimizer and the categorical cross-entropy loss function. The batch size is set to 128 and the model is trained for 10 epochs.

```
opt=keras.optimizers.Adam(learning_rate=0.001)
model1.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=["accuracy"])
```
### Evaluation
The model achieves an accuracy of around 99% on the test set. The evaluation is performed using the evaluate method of the model.
![image](https://user-images.githubusercontent.com/130960032/232430349-76424e5b-fd12-42ee-bf84-0501b581b006.png)
![image](https://user-images.githubusercontent.com/130960032/232430532-06f5e3da-833f-46a6-a2da-dc3c89a3613f.png)

# Usage

## Web-app
To use this web app, simply follow these steps:

* Open the deployed web app URL in your web browser.
* Upload an image file containing a handwritten digit that you would like to recognize.
* Click on the "Recognize Digit" button to initiate the digit recognition process.
* The predicted digit will be displayed on the web page, along with a visual representation of the input image and the corresponding probability distribution over all possible digit classes.

### Technical Details
The web app is built using the Streamlit Python library, which provides a simple and intuitive interface for creating interactive data visualization and machine learning applications. The app uses a pre-trained convolutional neural network (CNN) model to recognize handwritten digits from input images. The model was trained on the standard MNIST dataset of grayscale images of size 28x28 pixels, using a combination of convolutional, pooling, and fully connected layers. The model was optimized using the Adam optimizer and cross-entropy loss function, with early stopping based on validation accuracy to prevent overfitting.

The app is hosted on a cloud server using the Streamlit sharing service, which provides a convenient way to deploy and share Streamlit apps with others over the internet. The app is automatically deployed and updated whenever changes are made to the underlying code repository, and can be accessed using a web browser from anywhere with an internet connection. The app also includes various user interface elements, such as file upload buttons and progress bars, to enhance the user experience and make it easy to interact with the app.




## Conclusion
This project shows how to build a CNN for recognizing handwritten digits from the MNIST dataset using TensorFlow and Keras. The model achieves a high accuracy on the test set and can be used for various applications such as automated digit recognition in handwritten documents.
