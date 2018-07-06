#MIT License

#Copyright (c) 2018 Amir Aslan Haghrah

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import Image
import numpy as np
import tensorflow as tf

trainDataSetSize = 20000
trainDataLeft = 40
trainDataTop = 300
trainDataWidth = 320
trainDataHeight = 112
downSampleRate = 4


# Matrix which contains whole Test Gun images.
trainData = np.zeros(shape = (trainDataSetSize, int(trainDataWidth / downSampleRate * trainDataHeight / downSampleRate)))

# Manipulate 'Gun_Images_train_data_2D' with all Gun images.
for n in range(trainDataSetSize):
    image = np.asarray(Image.open('UnSupervisedData/' + str(n) + '.png').convert('L'))    # Loading images one by one to image buffer 
    image = (image > 199).astype(int)
    p = 0
    for i in range(np.shape(image)[0]):         # Trace Image Height
        for j in range(np.shape(image)[1]):     # Trace Image Width
            trainData[n][p] = image[i][j]
            p += 1

x = open("UnSupervisedData\imageLabel.txt" , "r")
label = np.zeros(shape = (trainDataSetSize, 3))
i = 0
for line in x:
    label[i] = [int(line[0]), int(line[1]), int(line[2])] 
    i += 1
    if(i == trainDataSetSize): break
x.close()

LayerOne = 200

# Create the model
x = tf.placeholder(tf.float32, [None, int(trainDataWidth / downSampleRate * trainDataHeight / downSampleRate)])
W = tf.Variable(tf.random_normal([int(trainDataWidth / downSampleRate * trainDataHeight / downSampleRate), LayerOne]))
b = tf.Variable(tf.random_normal([LayerOne]))
y = tf.nn.tanh(tf.matmul(x, W) + b)

WW = tf.Variable(tf.random_normal([LayerOne, 3]))
bb = tf.Variable(tf.random_normal([3]))
yy = tf.matmul(y, WW) + bb

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 3])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=yy))
train_step = tf.train.AdagradOptimizer(0.1).minimize(cross_entropy)

# Test trained model
correct_prediction = tf.equal(tf.argmax(yy, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Create a saver object which will save all the variables
saver = tf.train.Saver()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for k in range(2000):
    print(k)
    for i in range(int((trainDataSetSize / 200))):
        sess.run(train_step, feed_dict={x: trainData[i * 200: (i + 1) * 200], y_: label[i * 200: (i + 1) * 200]})

# Saving
saver.save(sess, "./model/TwoLayer_UnSupervised/TwoLayer_UnSupervised")

# Test
print(sess.run(accuracy, feed_dict={x: trainData, y_: label}))
sess.close()
