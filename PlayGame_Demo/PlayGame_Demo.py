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


import win32gui
import win32ui
import win32con
import win32api
import keyboard
import time
import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from win32api import GetSystemMetrics
from time import sleep


GameOver = False
endPlay = False

trainDataLeft = 40
trainDataTop = 300
trainDataWidth = 320
trainDataHeight = 112
downSampleRate = 4

# convert RGB image to Grayscale one
def rgb2gray(rgb):
    #return np.dot(rgb[...,:3], [0.299, 0.587, 0.114]) .astype('uint8')
    return np.dot(rgb[...,:3], [0.114, 0.587, 0.299]) .astype('uint8')

# Down sampling the input array by the rate of 'n'
def downSample(input , n):
    return input[::n,::n]

# List all of open windows.
toplist, winlist = [], []
def enum_cb(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))

# Mouse Click on specified location
def click(x,y):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)


# Click on specified locations to restart or start the game
def startGame():
    sleep(1)
    click(GameWindowLeft + int(GameWindowWidth / 2) - 37, GameWindowTop + GameWindowHeight - 58)
    sleep(1)
    click(GameWindowLeft + 360, GameWindowTop +  70)
    sleep(2)
    click(GameWindowLeft + int(GameWindowWidth / 2) - 37, GameWindowTop +  GameWindowHeight - 58)
    sleep(0.5)
    click(GameWindowLeft + int(GameWindowWidth / 2),GameWindowTop + int(GameWindowHeight / 2)) 
    sleep(1)

#######################################################################################################
# One layer Feed Forward Neural Network model with 200 neuran

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

#Create a saver object which will save all the variables
saver = tf.train.Saver()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver.restore(sess, "./model/TwoLayer_UnSupervised/TwoLayer_UnSupervised")

#######################################################################################################
# Capture images from the game window.
#######################################################################################################
win32gui.EnumWindows(enum_cb, toplist)

# Find the handle of the Game Window -> A Window with 'Ghost Light' title.
GameWindow = [(hwnd, title) for hwnd, title in winlist if 'ghost light' in title.lower()]
GameWindow = GameWindow[0]
hwnd = GameWindow[0]

win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, int(GetSystemMetrics(0) / 2) - 200, int(GetSystemMetrics(1) / 2) - 300, 400, 600, 0)

# Bring Game window to front and fucos on it.
win32gui.SetForegroundWindow(hwnd)

# Game Window rectangle dimensions
GameRect = win32gui.GetWindowRect(hwnd)
GameWindowLeft = GameRect[0]
GameWindowTop = GameRect[1]
GameWindowWidth = GameRect[2] - GameWindowLeft
GameWindowHeight = GameRect[3] - GameWindowTop

# grab a handle to the main desktop window
hdesktop = win32gui.GetDesktopWindow()

# create a device context
desktop_dc = win32gui.GetWindowDC(hdesktop)
img_dc = win32ui.CreateDCFromHandle(desktop_dc)
 
# create a memory based device context
mem_dc = img_dc.CreateCompatibleDC()

# create a bitmap object
screenshot = win32ui.CreateBitmap()
screenshot.CreateCompatibleBitmap(img_dc, GameWindowWidth, GameWindowHeight)
mem_dc.SelectObject(screenshot)

############################################################################################################
# start to play game, first capture image, then estimate the movement and then send specific key to the Game
############################################################################################################
while(True):
    # Reply
    input("Press Enter to continue...")
    startGame()
    GameOver = False        
    print('###############     ' + 'Reply : ' + '     ###############')
    sleep(1)  
    n = 0

    tmp = [1, 0, 1]
    leftKeyPressed = False
    rightKeyPressed = False

    while(not GameOver):       
        # copy the screen into our memory device context
        mem_dc.BitBlt((0, 0), (GameWindowWidth, GameWindowHeight), img_dc, (GameWindowLeft, GameWindowTop), win32con.SRCCOPY)
        # save the bitmap to a dataset
        signedIntsArray = screenshot.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype = 'uint8')
        img = img.reshape(GameWindowHeight, GameWindowWidth, 4)            

        trianInput = downSample(rgb2gray(img[trainDataTop:trainDataTop + trainDataHeight, trainDataLeft:trainDataLeft + trainDataWidth, :]), downSampleRate)           
        trianInput = (trianInput > 199).astype(int)
        tmp = yy.eval(feed_dict = {x: trianInput.reshape(1, trianInput.shape[0] * trianInput.shape[1])})

        if(np.argmin(tmp[0]) == 0):
            if (rightKeyPressed == True):
                keyboard.send('right', False, True)
                rightKeyPressed = False
                keyboard.send('left', True, False)
                leftKeyPressed = True
            elif (leftKeyPressed == False):
                keyboard.send('left', True, False)
                leftKeyPressed = True
        elif(np.argmin(tmp[0]) == 1):            
            if (leftKeyPressed == True):
                keyboard.send('left', False, True)
                leftKeyPressed = False
            if (rightKeyPressed == True):
                keyboard.send('right', False, True)
                rightKeyPressed = False 
        else: 
            if (leftKeyPressed == True):
                keyboard.send('left', False, True)
                leftKeyPressed = False
                keyboard.send('right', True, False)
                rightKeyPressed = True
            elif (rightKeyPressed == False):
                keyboard.send('right', True, False)
                rightKeyPressed = True

        #sleep(0.05)

        print(tmp, "\t" , np.argmin(tmp))
        n += 1

        if(all(img[72, 40] <= [240, 240, 240, 255])):
            GameOver = True           
            print('###############     ' + 'Game Over' + '     ###############')         
            print(tmp)
            print(np.argmin(tmp))    
            
            if (leftKeyPressed): print('left')
            elif (rightKeyPressed): print('right')
            else: print('center')
            break
                   
# free our objects
mem_dc.DeleteDC()
win32gui.DeleteObject(screenshot.GetHandle())