import pickle
import os.path

#graphic libraray
from tkinter import *
from tkinter import simpledialog

import numpy as np
import cv2 as cv
import PIL
import PIL.Image
import PIL.ImageDraw

#Ai models
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

class DrawingRecognizer:
    def __init__(self):
        self.class1, self.class2, self.class3 = None, None, None #will be the objects
        self.class1Counter, self.class2Counter, self.class3Counter = None, None, None #how many drawing are there in every object

        self.clf = None #ai model
        self.projectName = None #the name of the directory
        self.root = None #tkinter window
        self.image1 = None #playing with the image

        self.statusLabel = None #wich model is being used
        self.canvas = None
        self.draw = None #image draw object

        self.brushSize = 15 #defuelt brush size
        
        self.classesPrompt()
        self.initGui()
    
    #window messages
    def classesPrompt(self):
        msg = Tk()
        msg.withdraw()

        self.projectName = simpledialog.askstring("Project Name", "Enter project name", parent=msg)
        if os.path.exists(self.projectName):
            with open(f"{self.projectName}/{self.projectName}_data.pickle", "rb") as file:
                data = pickle.load(file)

            self.class1 = data['c1']
            self.class2 = data['c2']
            self.class3 = data['c3']
            self.class1Counter = data['c1c']
            self.class2Counter = data['c2c']
            self.class3Counter = data['c3c']
            self.clf = data['clf']
            self.projectName = data['pname']

        else:
            self.class1 = simpledialog.askstring("Object 1", "what is the first object?", parent = msg)
            self.class2 = simpledialog.askstring("Object 2", "what is the second object?", parent = msg)
            self.class3 = simpledialog.askstring("Object 3", "what is the third object?", parent = msg)

            self.class1Counter = 1
            self.class2Counter = 1
            self.class3Counter = 1

            self.clf = LinearSVC()

            os.mkdir(self.projectName)
            os.chdir(self.projectName)
            os.mkdir(self.class1)
            os.mkdir(self.class2)
            os.mkdir(self.class3)
            os.chdir("..")

    #graphic buttons
    def initGui(self):
        WIDTH = 500
        HEIGHT = 500
        WHITE = (255, 255, 255)

        self.root = Tk()
        self.root.title("Quick, Draw!")

        self.canvas = Canvas(self.root, width=WIDTH - 10 , height=HEIGHT- 10, bg="white")
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint())

        self.image1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = PIL.ImageDraw.Draw(self.image1)

        btnFrame = Frame(self.root)
        btnFrame.pack(fill = X, side = BOTTOM)
        
        btnFrame.columnconfigure(0, weight = 1)
        btnFrame.columnconfigure(2, weight = 1)
        btnFrame.columnconfigure(2, weight = 1)

        class1Btn = Button(btnFrame, text = self.class1, command= lambda: self.save(1))
        class1Btn.grid(row = 0, column= 0, sticky= W + E)

    def paint(self, event):
        pass

    def save(self):
        pass
    
