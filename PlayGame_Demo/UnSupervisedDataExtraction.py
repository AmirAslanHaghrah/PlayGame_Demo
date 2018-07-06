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
import matplotlib.pyplot as plt
from win32api import GetSystemMetrics
from time import sleep


GameOver = False
endPlay = False
trainDataSetSize = 20
trainData = [None] * trainDataSetSize
trainDataLeft = 40
trainDataTop = 300
trainDataWidth = 320
trainDataHeight = 112
downSampleRate = 4
playCount = 0
imageCount = 0



def rgb2gray(rgb):
    #return np.dot(rgb[...,:3], [0.299, 0.587, 0.114]) .astype('uint8')
    return np.dot(rgb[...,:3], [0.114, 0.587, 0.299]) .astype('uint8')

def downSample(input , n):
    return input[::n,::n]

# List all of open windows.
toplist, winlist = [], []
def enum_cb(hwnd, results):
    winlist.append((hwnd, win32gui.GetWindowText(hwnd)))

# Mouse Click in specified location
def click(x,y):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

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


imageLabel = open("./UnSupervisedData/imageLabel.txt", "w")

while(not endPlay and not keyboard.is_pressed('q') and playCount < 500):
    # Reply
    startGame()
    GameOver = False    
    print('###############     ' + 'Reply : ' + str(playCount) + '     ###############')
    sleep(2)  
    n = 0 

    # choose tmp randomly
    tmp = np.random.randint(3)

    while(not GameOver):
        if(keyboard.is_pressed('q')):
            endPlay = True
            break

        # copy the screen into our memory device context
        mem_dc.BitBlt((0, 0), (GameWindowWidth, GameWindowHeight), img_dc, (GameWindowLeft, GameWindowTop), win32con.SRCCOPY)
        # save the bitmap to a dataset
        signedIntsArray = screenshot.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype = 'uint8')
        img = img.reshape(GameWindowHeight, GameWindowWidth, 4)       

        if(all(img[72, 40] <= [240, 240, 240, 255])):
            GameOver = True           
            print('###############     ' + 'Game Over' + '     ###############')         
            break

        trianInput = downSample(rgb2gray(img[trainDataTop:trainDataTop + trainDataHeight, trainDataLeft:trainDataLeft + trainDataWidth, :]), downSampleRate)

        if(tmp == 0):
            pressedKey = [1, 0, 0]
            keyboard.release('right')
            keyboard.press('left')

        if(tmp == 1):
            pressedKey = [0, 1, 0]
            keyboard.release('left')
            keyboard.release('right')

        if(tmp == 2):
            pressedKey = [0, 0, 1]
            keyboard.release('left')
            keyboard.press('right') 

        trainData[n % trainDataSetSize] = trianInput
        n = n + 1                       
    
    playCount = playCount + 1
    for i in range(trainDataSetSize - ignoredTrainDataSize):
        img = Image.fromarray((trainData[(n % trainDataSetSize + i) % trainDataSetSize] * 255).astype('uint8'))
        img.save("./UnSupervisedData/" + str(imageCount) + ".png")
        imageLabel.write(str(pressedKey[0]))
        imageLabel.write(str(pressedKey[1]))
        imageLabel.write(str(pressedKey[2]))
        imageLabel.write("\n")
        imageCount = imageCount + 1
       
imageLabel.close()
        
# free our objects
mem_dc.DeleteDC()
win32gui.DeleteObject(screenshot.GetHandle())

