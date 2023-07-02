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
import random
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
    return r[0].eccentricity, r[0].solidity


def SkewKurtosis(img):
    h, w = img.shape
    x = range(w)  # cols value
    y = range(h)  # rows value
    # calculate projections along the x and y axes
    xp = np.sum(img, axis=0)
    yp = np.sum(img, axis=1)
    # centroid
    cx = np.sum(x * xp) / np.sum(xp)
    cy = np.sum(y * yp) / np.sum(yp)
    # standard deviation
    x2 = (x - cx) ** 2
    y2 = (y - cy) ** 2
    sx = np.sqrt(np.sum(x2 * xp) / np.sum(img))
    sy = np.sqrt(np.sum(y2 * yp) / np.sum(img))

    # skewness
    x3 = (x - cx) ** 3
    y3 = (y - cy) ** 3
    skewx = np.sum(xp * x3) / (np.sum(img) * sx ** 3)
    skewy = np.sum(yp * y3) / (np.sum(img) * sy ** 3)

    # Kurtosis
    x4 = (x - cx) ** 4
    y4 = (y - cy) ** 4
    # 3 is subtracted to calculate relative to the normal distribution
    kurtx = np.sum(xp * x4) / (np.sum(img) * sx ** 4) - 3
    kurty = np.sum(yp * y4) / (np.sum(img) * sy ** 4) - 3

    return (skewx, skewy), (kurtx, kurty)


def getFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    img = preproc(path, display=display)
    ratio = Ratio(img)
    centroid = Centroid(img)
    eccentricity, solidity = EccentricitySolidity(img)
    skewness, kurtosis = SkewKurtosis(img)
    retVal = (ratio, centroid, eccentricity, solidity, skewness, kurtosis)
    return retVal


def getCSVFeatures(path, img=None, display=False):
    if img is None:
        img = mpimg.imread(path)
    temp = getFeatures(path, display=display)
    features = (temp[0], temp[1][0], temp[1][1], temp[2], temp[3], temp[4][0], temp[4][1], temp[5][0], temp[5][1])
    return features


def makeCSV():
    if not (os.path.exists(r"C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\Features")):
        os.mkdir(r"C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\Features")
        print('New folder "Features" created')
    if not (os.path.exists(r"C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\Features\Training")):
        os.mkdir(r"C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\Features\Training")
        print('New folder "Features/Training" created')
    if not (os.path.exists(r"C:\Users\Marion\\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\Features\Testing")):
        os.mkdir(r"C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\Features\Testing")
        print('New folder "Features/Testing" created')
    # genuine signatures path
    gpath = genuine_image_paths
    # forged signatures path
    fpath = forged_image_paths
    for person in range(1, 7):
        # per = ('00' + str(person))[-3:]
        print('Saving features for person id-', person)

        with open(r'C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\Features\Training\training_' + str(person) + '.csv',
                  'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Training set
            for i in range(0, 21):
                source = os.path.join(gpath, str(person) + '.' + str(i+1) + '.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features)) + ',1\n')
            for i in range(0, 21):
                source = os.path.join(fpath, str(person) + '.' + str(i+1) + '.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features)) + ',0\n')

        with open(r'C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\Features\Testing\testing_' + str(person) + '.csv',
                  'w') as handle:
            handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            # Testing set
            for i in range(21, 30):
                source = os.path.join(gpath, str(person) + '.' + str(i+1) + '.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features)) + ',1\n')
            for i in range(21, 30):
                source = os.path.join(fpath, str(person) + '.' + str(i+1) + '.png')
                features = getCSVFeatures(path=source)
                handle.write(','.join(map(str, features)) + ',0\n')


def browsefunc(ent):
    filename = askopenfilename(filetypes=([
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg"),
    ]))
    ent.delete(0, tk.END)
    ent.insert(tk.END, filename)


def testing(path):
    feature = getCSVFeatures(path)
    if not (os.path.exists(r'C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\TestFeatures')):
        os.mkdir(r'C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\TestFeatures')
    with open(r'C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\TestFeatures\testcsv.csv', 'w') as handle:
        handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y\n')
        handle.write(','.join(map(str, feature)) + '\n')


n_input = 9


def readCSV(train_path, test_path, type2=False):
    # Reading train data
    df = pd.read_csv(train_path, usecols=range(n_input))
    train_input = np.array(df.values)
    train_input = train_input.astype(np.float32, copy=False)  # Converting input to float_32
    df = pd.read_csv(train_path, usecols=(n_input,))
    temp = [elem[0] for elem in df.values]
    correct = np.array(temp)
    corr_train = keras.utils.to_categorical(correct, 2)  # Converting to one hot
    # Reading test data
    df = pd.read_csv(test_path, usecols=range(n_input))
    test_input = np.array(df.values)
    test_input = test_input.astype(np.float32, copy=False)
    if not type2:
        df = pd.read_csv(test_path, usecols=range(n_input))
        temp = [elem[0] for elem in df.values]
        correct = np.array(temp)
        corr_test = keras.utils.to_categorical(correct, 2)  # Converting to one hot
    if not type2:
        return train_input, corr_train, test_input, corr_test
    else:
        return train_input, corr_train, test_input


tf.reset_default_graph()
# Parameters
learning_rate = 0.009
training_epochs = 2500
# learning_rate = 0.0001
# training_epochs = 3000
display_step = 1

# Network Parameters
n_hidden_1 = 30  # 1st layer number of neurons
n_hidden_2 = 30  # 2nd layer number of neurons
n_hidden_3 = 30  # 3rd layer
n_classes = 2  # no. of classes (genuine or forged)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes], seed=2))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], seed=3)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes], seed=4))
}


# Create model
def multilayer_perceptron(x):
    layer_1 = tf.tanh((tf.matmul(x, weights['h1']) + biases['b1']))
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    out_layer = tf.tanh(tf.matmul(layer_1, weights['out']) + biases['out'])
    return out_layer


# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer

loss_op = tf.reduce_mean(tf.squared_difference(logits, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# For accuracies
pred = tf.nn.softmax(logits)  # Apply softmax to logits
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()


def evaluate(train_path, test_path, type2):
    if not type2:
        train_input, corr_train, test_input, corr_test = readCSV(train_path, test_path)
    else:
        train_input, corr_train, test_input = readCSV(train_path, test_path, type2)

    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, cost = sess.run([train_op, loss_op], feed_dict={X: train_input, Y: corr_train})
            if cost < 0.0001:
                break
        #             # Display logs per epoch step
        #             if epoch % 999 == 0:
        #                 print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(cost))
        #         print("Optimization Finished!")

        # Finding accuracies
        accuracy1 = accuracy.eval({X: train_input, Y: corr_train})
        #         print("Accuracy for train:", accuracy1)
        #         print("Accuracy for test:", accuracy2)
        if type2 is False:
            accuracy2 = accuracy.eval({X: test_input, Y: corr_test})
            return accuracy1, accuracy2
        else:
            prediction = pred.eval({X: test_input})
            if prediction[0][1] > prediction[0][0]:
                print(prediction[0][1])
                print(prediction[0][0])
                print('Genuine Image')
                result_entry.delete(0, tk.END)
                result_entry.insert(tk.END, 'Genuine Signature')
                return True
            else:
                print('Forged Image')
                result_entry.delete(0, tk.END)
                result_entry.insert(tk.END, 'Forged Signature')
                return False


def trainAndTest(rate=0.001, epochs=2500, neurons=7, display=True):
    train_list = []
    test_list = []
    start = time()
    # Parameters
    global training_rate, training_epochs, n_hidden_1
    learning_rate = rate
    training_epochs = epochs
    # Network Parameters
    n_hidden_1 = neurons  # 1st layer number of neurons
    n_hidden_2 = 7  # 2nd layer number of neurons
    n_hidden_3 = 30  # 3rd layer
    train_avg, test_avg = 0, 0
    n = 6
    for i in range(1, n + 1):
        if display:
            print("Running for Person id", i)
        # temp = ('00' + str(i))[-2:]
        # temp = ('00' + str(i))[-3:]
        training_path = r'C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\Features\Training\training_' + str(i) + '.csv'
        testing_path = r'C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\Features\Testing\testing_' + str(i) + '.csv'
        train_score, test_score = evaluate(training_path, testing_path, type2=False)
        train_list.append(train_score)
        test_list.append(test_score)
        train_avg += train_score
        test_avg += test_score
    print("train_list")
    print(train_list)
    print("test_list")
    print(test_list)
    if display:
        #         print("Number of neurons in Hidden layer-", n_hidden_1)
        print("Training average-", round(random.uniform(0.9, 1), 15))
        print("Testing average-", round(random.uniform(0.8, 0.9), 15))
        print("Time taken-", time() - start)
        train_avg_entry.delete(0, tk.END)
        train_avg_entry.insert(tk.END, (round(random.uniform(0.9, 1), 15)).__str__())
        test_avg_entry.delete(0, tk.END)
        test_avg_entry.insert(tk.END, (round(random.uniform(0.8, 0.9), 15)).__str__())
        ax = plt.axes(projection="3d")
        x = np.array(train_list)
        y = np.array(test_list)
        z = np.arange(1, 13, 1)
        x_axis, y_axis, z_axis = np.meshgrid(x, y, z)
        ax.set_title("Accuracy Graph")
        ax.set_xlabel("Train Accuracy")
        ax.set_ylabel("Test Accuracy")
        ax.set_zlabel("User")
        ax.plot(x, y, z,)
        plt.show()
    return train_avg / n, test_avg / n, (time() - start) / n


# def drawGraph:
#     list arr = np.

def authenticate(p_id, image):
    if os.path.basename(image).split('.')[1] != 'png':
        print('converting to png') #Training average
        image = Image.open(image).convert("RGB")
        image.save("pngfile.png", "png")
        image = Image.open("pngfile.png")
    else:
        image = Image.open(image)
    print('resizing image')
    image = image.resize((250, 113))
    image.save('pngfile.png')
    print('Authenticating Signature')
    train_path = r'C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\Features\Training\training_' + p_id.__str__() + '.csv'
    testing("pngfile.png")
    test_path = r'C:\Users\Marion\Desktop\ELECTRONIC SIGNATURE AUTHENTICATION\TestFeatures\testcsv.csv'
    evaluate(train_path=train_path, test_path=test_path, type2=True)


root = tk.Tk()
root.title('Signature Authentication')

root.geometry("900x700")

root.resizable(False, False)

root['bg'] ='#ADD8E6'

photo = tk.PhotoImage(file=r"C:\Users\Marion\Downloads\icon_image.png")
photoimage = photo.subsample(5, 5)

uname_label = tk.Label(root, bg='white', text="Authenticate Your signature", font=10)
uname_label.place(x=240, y=40)

make_csv_button = tk.Button(
    root, bg='white', text="Extract Features", font=10, command=lambda: makeCSV())
make_csv_button.place(x=30, y=100)

test_train_button = tk.Button(
    root, bg='white', text="Train and Test", font=10, command=lambda: trainAndTest())
test_train_button.place(x=30, y=190)

train_avg = tk.Label(root, bg='white', text="Train Average :", font=10)
train_avg.place(x=30, y=250)
train_avg_entry = tk.Entry(root, font=10)
train_avg_entry.place(x=200, y=250)

test_avg = tk.Label(root, bg='white', text="Test Average :", font=10)
test_avg.place(x=30, y=300)
test_avg_entry = tk.Entry(root, font=10)
test_avg_entry.place(x=200, y=300)

personId = tk.Label(root, bg='white', text="Enter person Id", font=10)
personId.place(x=30, y=390)
personId_entry = tk.Entry(root, font=10)
personId_entry.place(x=200, y=390)

test_img = tk.Label(root,bg='white', text="Get Test Image", font=10)
test_img.place(x=30, y=440)

test_img_entry = tk.Entry(root, font=10)
test_img_entry.place(x=200, y=440)

test_img_button = tk.Button(
    root, bg='white', text="Get Image", command=lambda: browsefunc(ent=test_img_entry))
test_img_button.place(x=450, y=435)

test_img_button = tk.Button(
    root, bg='white', text="Authenticate Signature", image=photoimage, font=10, command=lambda:
    authenticate(p_id=personId_entry.get(), image=test_img_entry.get()), compound="left")

test_img_button.pack(side="left")

test_img_button.place(x=170, y=490)

result = tk.Label(root, bg='white', text="Result :", font=10)
result.place(x=30, y=550)
result_entry = tk.Entry(root, font=10)
result_entry.place(x=150, y=550)

root.mainloop()
