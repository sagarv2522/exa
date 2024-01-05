# ml_app.py
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train the model
model = SVC()
model.fit(X_train, y_train)

# Create a Streamlit web app
st.title("Simple ML Web App")

# Add a sidebar for user input
st.sidebar.header("User Input")

# Collect user input for prediction
sepal_length = st.sidebar.slider("Sepal Length", float(X_train[:, 0].min()), float(X_train[:, 0].max()), float(X_train[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width", float(X_train[:, 1].min()), float(X_train[:, 1].max()), float(X_train[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length", float(X_train[:, 2].min()), float(X_train[:, 2].max()), float(X_train[:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width", float(X_train[:, 3].min()), float(X_train[:, 3].max()), float(X_train[:, 3].mean()))

# Make predictions
user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(user_input)

# Display the prediction and accuracy
st.write(f"Prediction: {iris.target_names[prediction[0]]}")
st.write(f"Model Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")
