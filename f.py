import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import tkinter as tk
from tkinter import ttk

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import torch
import torch.nn as nn
import torch.optim as optim

# Function to load and preprocess the face recognition model
def load_face_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    face_model = Model(inputs=base_model.input, outputs=x)
    return face_model

# Function to load and preprocess the regression model using ElasticNet
def load_regression_model(X_train, y_train):
    model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to load and preprocess the neural network regression model using PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

def load_neural_network(X_train, y_train):
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1000):
        inputs = torch.tensor(X_train.values, dtype=torch.float32)
        targets = torch.tensor(y_train.values, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    return model

# Function to perform face recognition using TensorFlow
def recognize_face(img, face_model):
    imgS = cv2.resize(img, (224, 224))
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    img_array = image.img_to_array(imgS)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    face_encoding = face_model.predict(img_array)
    return face_encoding.flatten()

# Function to mark attendance
def mark_attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# Function to create GUI for attendance marking
def create_gui(name, window):
    def on_mark_attendance():
        mark_attendance(name)
        messagebox.showinfo("Attendance Marked", f"Attendance for {name} marked successfully!")

    mark_button = ttk.Button(window, text="Mark Attendance", command=on_mark_attendance)
    mark_button.pack()

# Load data for regression (dummy data for demonstration purposes)
data = pd.read_csv("your_regression_data.csv")
y_regression = data.pop("target_column")
X_regression_train, X_regression_test, y_regression_train, y_regression_test = train_test_split(data, y_regression, test_size=0.2, random_state=42)

# Load models
face_model = load_face_model()
elastic_net_model = load_regression_model(X_regression_train, y_regression_train)
neural_network_model = load_neural_network(X_regression_train, y_regression_train)

# Main video capture loop
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    # Perform face recognition
    face_encoding = recognize_face(img, face_model)

    # Perform regression prediction using ElasticNet
    regression_prediction_en = elastic_net_model.predict([face_encoding])[0]

    # Perform regression prediction using Neural Network
    X_regression_nn = pd.DataFrame([face_encoding])
    regression_prediction_nn = neural_network_model(torch.tensor(X_regression_nn.values, dtype=torch.float32)).detach().numpy()[0][0]

    # Display GUI for marking attendance
    name = "Dummy Name"  # Replace with your logic to get the actual name
    root = tk.Tk()
    root.title("Attendance Marking")

    ttk.Label(root, text=f"ElasticNet Regression Prediction: {regression_prediction_en}").pack()
    ttk.Label(root, text=f"Neural Network Regression Prediction: {regression_prediction_nn}").pack()

    create_gui(name, root)

    root.mainloop()

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
