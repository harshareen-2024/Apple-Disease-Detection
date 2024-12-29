**Apple Disease Detection and Prediction System**

This repository contains an Apple Disease Detection and Prediction System designed to assist farmers and agricultural experts in identifying apple diseases, predicting apple quality, and forecasting apple stock prices. The system uses machine learning and deep learning models to detect diseases in apples and apple leaves, evaluate fruit quality, and predict future apple prices.

Additionally, the system includes an alarm feature that triggers a sound when a diseased apple or leaf is detected, helping farmers take early action to prevent further damage.

**Key Features**

**Fruit Disease Detection**: Identifies and classifies diseases in apples (e.g., Blotch Apple, Rot Apple, Scab Apple, Normal Apple) using a trained deep learning model.

**Leaf Disease Detection:** Detects diseases in apple leaves (e.g., Rust, Scab, Powdery Mildew) using a TensorFlow Lite model.

**Apple Quality Prediction:** Classifies apples as "Good" or "Bad" based on attributes such as size, weight, sweetness, crunchiness, juiciness, ripeness, and acidity.

**Stock Price Prediction:** Predicts the future apple stock price using historical data and machine learning models.

**Fertilizer Recommendations:** Provides fertilizer suggestions based on the detected diseases for both apple fruits and leaves.

**Alarm System:** Triggers a sound alarm when a diseased apple or leaf is detected, helping farmers take early action before the situation worsens.

**Technologies Used**

Programming Language: Python
Framework: Flask

**Libraries:**

TensorFlow/Keras (for deep learning models)
OpenCV (for image processing)
Pandas (for data manipulation)
scikit-learn (for stock price prediction)
Pillow (for image handling)
Winsound (for triggering alarms on Windows)

**Model Types:**

CNN (Convolutional Neural Networks) for fruit disease detection.
TensorFlow Lite for efficient leaf disease detection.
Regression models for stock price prediction.
Web Interface: HTML, CSS, JavaScript (via Flask routes)

**How It Works**

**Fruit Disease Detection:**

Upload an image of an apple, and the system predicts whether it is diseased (Blotch Apple, Rot Apple, etc.) or healthy.
Provides relevant fertilizer suggestions based on the detected disease.
Alarm System: If a diseased apple is detected, an alarm sound is triggered, alerting the farmer to take immediate action.

**Leaf Disease Detection:**

Upload an image of an apple leaf, and the system detects diseases such as Rust, Scab, or Powdery Mildew.
Provides fertilizer recommendations.

**Alarm System:** If a diseased leaf is detected, an alarm is triggered to help farmers respond early.

**Apple Quality Prediction:**

Submit attributes like size, weight, sweetness, and crunchiness of the apple to predict whether it is of "Good" or "Bad" quality.
Apple Stock Price Prediction:

Enter a future date, and the system predicts the stock price for apples based on historical data.

**Dataset**

Apple Disease Dataset: The dataset used for training the apple disease detection models consists of labeled images of apples, including categories like healthy apples and various disease types.
Apple Stock Price Data: Historical apple price data (Modal Price per Quintal) is used to predict future stock prices.

**Contributions**
Feel free to contribute to this project! Open an issue or submit a pull request for any improvements.
