import streamlit as st
import numpy as np
import h5py

# Load dataset
train_dataset = h5py.File(r'F:\vs code\train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:])

# Streamlit app
st.title("Dataset Viewer")

# Input box for index
index = st.number_input("Enter the index (between 0 and {})".format(train_set_x_orig.shape[0] - 1), value=0, min_value=0, max_value=train_set_x_orig.shape[0] - 1)

# Display matrix for the specified index
st.write("Matrix for index {}: ".format(index))
st.write(train_set_x_orig[index])
