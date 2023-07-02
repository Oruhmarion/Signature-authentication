import tkinter as tk
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename
import tensorflow._api.v2.compat.v1 as tf
import pandas as pd
import numpy as np
from time import time
import keras
from skimage.filters import threshold_otsu
from scipy import ndimage
from skimage.measure import regionprops
import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from mpl_toolkits import mplot3d
from tkinter import messagebox
import customtkinter


tf.disable_v2_behavior()

genuine_image_paths = r"C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\ELECRONIC SIGNATURE"
forged_image_paths = r"C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\ELECTRONIC FORGERY"

def rgbgrey(img):
    # Converts rgb to grayscale
    greyimg = np.zeros((img.shape[0], img.shape[1]))
    for row in range(len(img)):
        for col in range(len(img[row])):
            greyimg[row][col] = np.average(img[row][col])
    return greyimg


def greybin(img):
    # Converts grayscale to binary
    blur_radius = 0.8
    img = ndimage.gaussian_filter(img, blur_radius)  # to remove small components or noise
    #     img = ndimage.binary_erosion(img).astype(img.dtype)
    thres = threshold_otsu(img)
    binimg = img > thres
    binimg = np.logical_not(binimg)
    return binimg


def preproc(path, img=None, display=True):
    if img is None:
        img = mpimg.imread(path)
    if display:
        plt.imshow(img)
        plt.show()
    grey = rgbgrey(img)  # rgb to grey
    if display:
        plt.imshow(grey, cmap=matplotlib.cm.Greys_r)
        plt.show()
    binimg = greybin(grey)  # grey to binary
    if display:
        plt.imshow(binimg, cmap=matplotlib.cm.Greys_r)
        plt.show()
    r, c = np.where(binimg == 1)
    # Now we will make a bounding box with the boundary as the position of pixels on extreme.
    # Thus we will get a cropped image with only the signature part.
    signimg = binimg[r.min(): r.max(), c.min(): c.max()]
    if display:
        plt.imshow(signimg, cmap=matplotlib.cm.Greys_r)
        plt.show()
    return signimg


def Ratio(img):
    a = 0
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col] == True:
                a = a + 1
    total = img.shape[0] * img.shape[1]
    return a / total


def Centroid(img):
    numOfWhites = 0
    a = np.array([0, 0])
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col] == True:
                b = np.array([row, col])
                a = np.add(a, b)
                numOfWhites += 1
    rowcols = np.array([img.shape[0], img.shape[1]])
    centroid = a / numOfWhites
    centroid = centroid / rowcols
    return centroid[0], centroid[1]


def EccentricitySolidity(img):
    r = regionprops(img.astype("int8"))
    return r

[0].eccentricity, r[0].solidity


def SkewKurt(img):
    if np.sum(img) != 0:
        M = cv2.moments(img)
        skew = round(float(M['mu11']) / float(M['mu02']), 2)
        kurt = round(float(M['mu40']) / float(M['mu02']), 2)
        return skew, kurt
    else:
        return 0, 0


def extract_features(image_path):
    img = preproc(image_path, img=None, display=False)
    ratio = Ratio(img)
    centroid_x, centroid_y = Centroid(img)
    eccentricity, solidity = EccentricitySolidity(img)
    skewness, kurtosis = SkewKurt(img)
    features = [ratio, centroid_x, centroid_y, eccentricity, solidity, skewness, kurtosis]
    return features


def create_dataset(genuine_paths, forged_paths):
    data = []
    labels = []
    for path in genuine_paths:
        features = extract_features(path)
        data.append(features)
        labels.append(1)  # genuine class label: 1
    for path in forged_paths:
        features = extract_features(path)
        data.append(features)
        labels.append(0)  # forged class label: 0
    return data, labels


def normalize_data(data):
    data = np.array(data)
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))
    return data


def train_test_split(data, labels, test_size=0.2):
    num_samples = len(data)
    num_test_samples = int(test_size * num_samples)
    indices = np.random.permutation(num_samples)
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]
    train_data = data[train_indices]
    train_labels = np.array(labels)[train_indices]
    test_data = data[test_indices]
    test_labels = np.array(labels)[test_indices]
    return train_data, train_labels, test_data, test_labels


def create_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(100, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_model(train_data, train_labels, input_shape, epochs=100):
    model = create_model(input_shape)
    model.fit(train_data, train_labels, epochs=epochs, verbose=0)
    return model


def evaluate_model(model, test_data, test_labels):
    _, accuracy = model.evaluate(test_data, test_labels, verbose=0)
    return accuracy


def predict(model, data):
    predictions = model.predict(data)
    return predictions.flatten()


def get_prediction_label(prediction):
    if prediction >= 0.5:
        return "Genuine"
    else:
        return "Forged"


def open_file_dialog():
    filename = askopenfilename(filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")))
    if filename:
        img = Image.open(filename)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_panel.configure(image=img)
        image_panel.image = img
        predict_button.configure(state="normal")
        clear_button.configure(state="normal")
        filepath_entry.delete(0,

 tk.END)
        filepath_entry.insert(0, filename)


def clear_image():
    image_panel.configure(image="")
    predict_button.configure(state="disabled")
    clear_button.configure(state="disabled")
    filepath_entry.delete(0, tk.END)


def predict_signature():
    filepath = filepath_entry.get()
    if os.path.exists(filepath):
        features = extract_features(filepath)
        features = normalize_data([features])
        prediction = predict(model, features)
        label = get_prediction_label(prediction[0])
        messagebox.showinfo("Prediction", f"The signature is {label}.")
    else:
        messagebox.showerror("Error", "Invalid file path.")


genuine_image_paths = [os.path.join(genuine_image_paths, filename) for filename in os.listdir(genuine_image_paths)]
forged_image_paths = [os.path.join(forged_image_paths, filename) for filename in os.listdir(forged_image_paths)]

data, labels = create_dataset(genuine_image_paths, forged_image_paths)
data = normalize_data(data)
input_shape = (len(data[0]),)

train_data, train_labels, test_data, test_labels = train_test_split(data, labels)
model = train_model(train_data, train_labels, input_shape)
accuracy = evaluate_model(model, test_data, test_labels)
print("Model Accuracy:", accuracy)

root = tk.Tk()
root.title("Signature Authenticator")
root.geometry("400x300")

filepath_label = tk.Label(root, text="File Path:")
filepath_label.pack()

filepath_entry = tk.Entry(root, width=30)
filepath_entry.pack()

browse_button = tk.Button(root, text="Browse", command=open_file_dialog)
browse_button.pack()

image_panel = tk.Label(root)
image_panel.pack()

predict_button = tk.Button(root, text="Predict", command=predict_signature, state="disabled")
predict_button.pack()

clear_button = tk.Button(root, text="Clear", command=clear_image, state="disabled")
clear_button.pack()

root.mainloop()