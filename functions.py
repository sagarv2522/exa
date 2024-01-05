# app.py
import streamlit as st
import numpy as np
from PIL import Image
import h5py

def load_dataset():
    ''' 
    This Function is used to read the h5py file to normal np array formate and decode the file formate

    Args: None
    Return: np arrays of required feature

    rewrite the code accordingly with replacement of file path of data
    '''
    
    # load the train dataset as h5 file
    train_dataset = h5py.File('train_catvnoncat.h5', "r")  # change the file path accordingly
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    # load the test dataset as h5 file
    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    # extract the list of classes
    classes = np.array(test_dataset["list_classes"][:])

    # reshape the dimension of y set
    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    assert train_set_y.shape == (1, train_set_y_orig.shape[0])
    assert test_set_y.shape == (1, test_set_y_orig.shape[0])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def getting_data():
    # Loading the data (cat/non-cat)
    train_X_org, train_Y, test_X_org, test_Y, classes = load_dataset()

    # reshaping the image related to dimension
    train_x_flatten = train_X_org.reshape(train_X_org.shape[0], -1).T
    test_x_flatten = test_X_org.reshape(test_X_org.shape[0], -1).T

    assert train_x_flatten.shape != train_X_org.shape
    assert test_x_flatten.shape != test_X_org.shape

    # standardize the image with 255
    train_X = train_x_flatten / 255
    test_X = test_x_flatten / 255

    return train_X, train_Y, test_X, test_Y, classes



st.title("Image Dataset Viewer")
st.write("Select the dataset (Train or Test), input an index, and view the corresponding image.")

dataset_option = st.selectbox("Select Dataset", ["Train", "Test"])
index = st.number_input("Enter Index:", min_value=0, value=0)

if dataset_option == "Train":
    data, labels, _, _, _ = getting_data()
else:
    _, _, data, labels, _ = getting_data()

if 0 <= index < data.shape[1]:
    st.image(data[:, index].reshape(64, 64, 3), caption=f"Label: {labels[0, index]}", use_column_width=True)
else:
    st.warning("Invalid Index. Please enter a valid index within the dataset size.")
