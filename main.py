#General libraries
import json
import numpy as np

#Graphics libraries
import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox

#Image reading libraries
import cv2 as cv
import PIL
from PIL import Image, ImageDraw

#CNN model libraries
from tensorflow import keras
from keras import layers, models
from keras.constraints import MaxNorm

class DrawingBoardGraphics:
    #this class keeps all the graphic code (drawing methods, buttons...)
    def __init__(self):
        self.WIDTH = 500 #the width of the window
        self.HEIGHT = 500 #the height of the window
        self.COLOR_WHITE = (255, 255, 255) #the color white by RGB

        self.root = tk.Tk() #the backend of the drawing canvas
        self.root.title("Drawing classifier")

        self.canvas = tk.Canvas(self.root, width = self.WIDTH, height = self.HEIGHT, bg = "white") #creating the window
        self.canvas.pack(expand = tk.YES, fill = tk.BOTH) #button widgets
        self.canvas.bind("<B1-Motion>", self.paint) #drawing canvas
        
        #a temp image file for every current drawing in the canvas
        self.currentImage = Image.new("RGB", (self.WIDTH, self.HEIGHT), self.COLOR_WHITE)
        self.draw = ImageDraw.Draw(self.currentImage)

        self.buttons()

    def buttons(self):
        btnFrame = tk.Frame(self.root)
        btnFrame.pack(fill = tk.X, side = tk.BOTTOM)

        btnFrame.columnconfigure(0, weight = 1)
        btnFrame.columnconfigure(1, weight = 1)
        btnFrame.columnconfigure(2, weight = 1)

    def paint(self):
        pass
