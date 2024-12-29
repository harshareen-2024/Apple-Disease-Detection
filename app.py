from datetime import datetime,timedelta
import io
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import TimeSeriesSplit
import cv2
import base64
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from data.apple_fruit_data import apple_fruit_data
from data.apple_leaf_data  import apple_leaf_data
from data.fertilizer_links import fertilizer_links
import os

app = Flask(__name__)


# Load trained model
fruit_model = load_model('models/fruit_model_best_weight.hdf5')

#Load fruit quality model
quality_model = load_model("models/apple_quality_detection_model.hdf5")

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="models/converted_model.tflite")
interpreter.allocate_tensors()
# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Load stock model
stock_model=load_model("models/apple-stock-model.keras")

# Load the historical data
apple_data = pd.read_csv("data/Apple.csv", index_col="Price Date", parse_dates=["Price Date"])
apple_data['Modal Price (Rs./Quintal)'] = apple_data['Modal Price (Rs./Quintal)'].apply(lambda x: (x+9000)/100)

# Sample fertilizer_links


def fruit_preprocess_image(image_data):
    try:
        # Decode base64 image data
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize the image to match the input shape of your model
        resized_image = cv2.resize(image, (224, 224))

        # Normalize the image pixel values to be between 0 and 1
        normalized_image = resized_image / 255.0

        # Expand the dimensions of the image to match the batch size (1 in this case)
        processed_image = np.expand_dims(normalized_image, axis=0)

        # Debugging print statements
        print("Processed Image Shape:", processed_image.shape)

        return processed_image
    except Exception as e:
        print("Error processing image:", e)
        return None


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the POST request
        image_data = request.json['image']

        # Preprocess the image
        processed_image = fruit_preprocess_image(image_data)

        if processed_image is None:
            return jsonify({'error': 'Error processing image'})

        # Make prediction using your trained model
        prediction = fruit_model.predict(processed_image)
        print(prediction)

        # Convert prediction to human-readable format
        class_names = ['Blotch Apple', 'Normal Apple', 'Rot Apple', 'Scab Apple']
        predicted_class = class_names[np.argmax(prediction)]
        if predicted_class in fertilizer_links:
            return jsonify({'prediction': predicted_class,'fertilizer':fertilizer_links[predicted_class]})
        else:
            return jsonify({'prediction': predicted_class, 'fertilizer': None})
    except Exception as e:
        # Return error message as JSON
        return jsonify({'error': str(e)})

def preprocess_leaf_image(image):
    # Resize the image to match the input shape of the model
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the pixel values
    image = (image.astype(np.float32) / 255.0)[np.newaxis, :]
    return image

def predict_leaf_disease(image):
    # Preprocess the image
    image = preprocess_leaf_image(image)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    # Define the classes for prediction
    classes=['complex', 'frog_eye_leaf_spot', 'healthy', 'powdery_mildew', 'rust','scab']
    # Get the predicted class
    predicted_class = classes[np.argmax(output)]
    

    return predicted_class

@app.route('/leaf_predict',methods=['POST'])
def leaf_predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    # Convert base64 image data to image
    image_data = request.json['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))

    # Predict the disease
    prediction = predict_leaf_disease(image)
    if prediction in fertilizer_links:
        return jsonify({'prediction': prediction,'fertilizer':fertilizer_links[prediction]})
    else:
        return jsonify({'prediction': prediction, 'fertilizer': None})

@app.route('/quality_predict',methods=['POST'])
def qaulity_predict():
    # Get input values from the request
    size = float(request.form['size'])
    weight = float(request.form['weight'])
    sweetness = float(request.form['sweetness'])
    crunchiness = float(request.form['crunchiness'])
    juiciness = float(request.form['juiciness'])
    ripeness = float(request.form['ripeness'])
    acidity = float(request.form['acidity'])
    target=0

    input_data = np.array([[acidity,ripeness,juiciness,crunchiness,sweetness, weight,size,target]])
    prediction = quality_model.predict(np.array(input_data).reshape(1, -1))
    # Scale the input values

    print(prediction[0][0])
    # Return the prediction as a JSON response
    if(prediction[0][0]>=0.5):
        return jsonify({'prediction': 'Good'})
    else:
        return jsonify({'prediction': 'Bad'})
        
@app.route('/predict_stock_price', methods=['POST'])
def predict_stock_price():
    # Get the future date as user input
    future_date = request.json['future_date']

    # Convert the future date to datetime format
    future_date = pd.to_datetime(future_date)

    # Scale the testing set
    scaler = MinMaxScaler()

    # Scale the last 60 days of data
    if future_date in apple_data.index:
        # Find the index of the future_date in the dataset
        idx = apple_data.index.get_loc(future_date)
        # Select the last 60 records starting from the future_date
        last_60_days_scaled = scaler.fit_transform(apple_data[['Modal Price (Rs./Quintal)']].iloc[idx-59:idx+1])
    else:
        # Scale the last 60 days of data
        last_60_days_scaled = scaler.fit_transform(apple_data[['Modal Price (Rs./Quintal)']].tail(60))

    # Reshape the last 60 days of data
    X_test = np.array(last_60_days_scaled).reshape(60, 1)

    # Predict the stock price for the future date
    y_pred = stock_model.predict(X_test)

    # Inverse transform the predicted price
    predicted_price = scaler.inverse_transform(y_pred)
    print(predicted_price)
    # Convert the predicted price to a serializable format (float)
    # predicted_price = float(predicted_price)
    # print(predicted_price)
    # Return the predicted stock price as a response
    return jsonify({'prediction':  predicted_price.flatten().tolist()})

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Fruit disease route
@app.route('/fruit_disease_prediction')
def fruit_disease_prediction():
    return render_template('fruit_disease_prediction.html',apple_data=apple_fruit_data,fertizer_data=fertilizer_links)

# Leaf detection route
@app.route('/leaf_disease_prediction')
def leaf_disease_prediction():
    return render_template('leaf_disease_prediction.html',apple_data=apple_leaf_data)

# Quality detection route
@app.route('/fruit_quality_detection')
def fruit_quality_detection():
    return render_template('fruit_quality_detection.html')

#stock price prediction
@app.route('/stock_price_prediction')
def stock_price_prediction():
    return render_template('stock_price_prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
