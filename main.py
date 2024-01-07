#General libraries
import os
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
        self.model = DrawingClassifeirCnnModel() #loadind the model

        self.WIDTH = 500 #the width of the window
        self.HEIGHT = 500 #the height of the window
        self.COLOR_WHITE = (255, 255, 255) #the color white by RGB
        self.bruhWidth = 15 #the size of the brush

        self.root = tk.Tk() #the backend of the drawing canvas
        self.root.title("Drawing classifier")

        self.canvas = tk.Canvas(self.root, width = self.WIDTH, height = self.HEIGHT, bg = "white") #creating the window
        self.canvas.pack(expand = tk.YES, fill = tk.BOTH) #button widgets
        self.canvas.bind("<B1-Motion>", self.paint) #drawing canvas
        
        #a temp image file for every current drawing in the canvas
        self.currentImage = Image.new("RGB", (self.WIDTH, self.HEIGHT), self.COLOR_WHITE)
        self.draw = ImageDraw.Draw(self.currentImage)

        self.buttons()

        self.root.protocol("WM_DELETE_WINDOW", self.onClose)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    #a function that creates the buttons in the bottom of the drawing canvas
    def buttons(self):
        btnFrame = tk.Frame(self.root)
        btnFrame.pack(fill = tk.X, side = tk.BOTTOM)

        btnFrame.columnconfigure(0, weight = 1)
        btnFrame.columnconfigure(1, weight = 1)
        btnFrame.columnconfigure(2, weight = 1)

        clearBtn = tk.Button(btnFrame, text = "Clear Drawing", command = self.clearDrawing)
        clearBtn.grid(row = 0, column = 0, sticky = tk.W + tk.E)
        clearBtn.config(font = ("Century Gothic", 12))

        saveDrawingBtn = tk.Button(btnFrame, text = "Save Drawing", command = lambda: self.saveDrawing())
        saveDrawingBtn.grid(row = 0, column = 1, sticky = tk.W + tk.E)
        saveDrawingBtn.config(font = ("Century Gothic", 12))

        trainModelBtn = tk.Button(btnFrame, text = "Train Model", command = self.model.trainModel)
        trainModelBtn.grid(row = 0, column = 2, sticky = tk.W + tk.E)
        trainModelBtn.config(font = ("Century Gothic", 12))

        predictDrawing = tk.Button(btnFrame, text = "Predict Drawing", command = self.model.predictDrawing)
        predictDrawing.grid(row = 1, column = 1, sticky = tk.W + tk.E)
        predictDrawing.config(font = ("Century Gothic", 12))

        self.statusLabel = tk.Label(btnFrame, text = "Haha I'm smart ROBOT")
        self.statusLabel.config(font = ("Century Gothic", 10))
        self.statusLabel.grid(row = 2, column = 1, sticky = tk.W + tk.E)

    #a function that paints with black pixels in the drawing canvas
    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)

        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.bruhWidth)
        self.draw.rectangle([x1, y2, x2 + self.bruhWidth, y2 + self.bruhWidth], fill="black", width=self.bruhWidth)

    #a function that clears the drawing board
    def clearDrawing(self):
        self.canvas.delete("all") #deleting everything from the drawing canvas

        self.draw.rectangle([0, 0, 1000, 1000], fill = "white")#making sure everything is cleared by drawing the canvas with white
    
    #a function that closes the window when clicked on close, and saving the model
    def onClose(self):
        with open("QuickDrawData\\ObjectsList.json", "w") as data:
            json.dump(self.model.dataList, data)

        self.model.saveModel() #saving the model
        self.root.destroy() #closing the window
        exit() #stoping the program

    def saveDrawing(self):
        '''a function that saves the model based of its name- 
        asking the user to name the drawing. 
        if the object exists in the data, the function will save it in its directory
         if not, creating a new directory and counter '''

        objectName = simpledialog.askstring("Drawing saving", "What is the object in the drawing?", parent = self.root)

        self.currentImage.save("QuickDrawData/currentDrawing.png") #saving the current drawing
        drawing = PIL.Image.open("QuickDrawData/currentDrawing.png") #opening it with PIL
        drawing.thumbnail((50, 50), PIL.Image.Resampling.LANCZOS)

        if objectName in self.model.dataList: #if the object exists in the data
            self.model.dataList[objectName] += 1 #increasing the counter
            drawing.save(f"QuickDrawData/{objectName}/{self.model.dataList[objectName]}.png", "PNG")#saving the image in the data

        else:
            self.model.dataList[objectName] = 1 #creating new object in the data file
            os.mkdir(f"QuickDrawData/{objectName}")

            drawing.save(f"QuickDrawData/{objectName}/1.png", "PNG")
        
class DrawingClassifeirCnnModel:
    def __init__(self):
        #open the json file with all the names of the current objects
        with open("QuickDrawData\\ObjectsList.json", "r") as data:
            self.dataList = json.load(data)

        self.dataLength = len(self.dataList)#the number of objects in the data

        self.root = tk.Tk()

        #creating the model
        self.model = keras.Sequential([
            layers.Conv2D(32, (3, 3), input_shape = (50, 50, 1), padding = 'same', activation = 'relu', kernel_constraint = MaxNorm(3)),
            layers.Dropout(0.2),
            layers.Conv2D(32, (3, 3), activation = 'relu', padding = 'same', kernel_constraint = MaxNorm(3)),
            layers.MaxPooling2D(pool_size = (2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation = 'relu', kernel_constraint = MaxNorm(3)),
            layers.Dropout(0.5),
            layers.Dense(self.dataLength, activation = 'softmax')
        ])

        self.model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    #a function to train the model
    def trainModel(self):
        #saving the data, to make sure it saved
        with open("QuickDrawData\\ObjectsList.json", "w") as data:
            json.dump(self.dataList, data)

        drawingsList = [] #the list with all the drawings in the data
        objectList = [] #the list with the objects in the data

        labelEncoder = {objectName: label for label, objectName in enumerate(self.dataList.keys())}

        for objectName, objectCounter in self.dataList.items(): #a loop for all the drawings
            for drawingNum in range(1, objectCounter): #a loop for every object seperately
                drawingPath = f"QuickDrawData/{objectName}/{drawingNum}.png" #saving the path to the drawing
                drawing = cv.imread(drawingPath, cv.IMREAD_GRAYSCALE) #reading the drawing in low quiality

                if drawing is not None: #making sure the drawing exists
                    drawing = cv.resize(drawing, (50, 50)) #resizing the drawing
                    drawing = drawing.reshape((50, 50, 1))
                    drawingsList.append(drawing) #inserting it into the list
                    objectList.append(labelEncoder[objectName])
                
                else:
                    print(f"Eror loading drawing {drawingPath}")

        drawingArray = np.array(drawingsList) #converting the array to numpy array
        objectArray = np.array(objectList) #converting the array to numpy array

        drawingArray = drawingArray / 255.0 #resizing the array again

        self.model.fit(drawingArray, objectArray, epochs = 20) #training the model

        messagebox.showinfo("Drawing Trainer", "Model successfully trained!", parent = self.root)

    #the function that predicts the current drawing
    def predictDrawing(self):
        drawingToPredict = Image.new("RGB", (500, 500), (255, 255, 255))
        
        # Save the current drawing to a temporary file
        drawingToPredict.save("QuickDraw/toPredict.png")
        drawing  = cv.imread("QuickDraw/toPredict.png", cv.IMREAD_GRAYSCALE)
        drawing = cv.resize(drawing, (50, 50))
        drawing = drawing.reshape((1, 50, 50, 1))

        drawing = drawing / 255.0

        #make prediction using the model
        prediction = self.model.predict(drawing)[0]

        objectPredict = {objectName: prediction[idx] * 100 for idx, objectName in enumerate(self.dataList.keys())}

        sortedDrawings = sorted(objectPredict.keys(), key=lambda x: objectPredict[x], reverse=True)
        
        threshold = 0.7
        topThreeGuesses = [(class_name, objectPredict[class_name]) for class_name in sortedDrawings[:3] if objectPredict[class_name] >= threshold]

        message = "I think the drawing is: \n"
        for i, (predictedObject, confidence) in enumerate(topThreeGuesses):
            message += f"{i+1}. {predictedObject} with {confidence:.2f}% confidence\n"
        
        isCorrect = messagebox.askyesno("Predicted Object", f"{message}Am I right?", parent=self.root)

        if isCorrect is not None:
            if isCorrect:
                self.saveDrawing(topThreeGuesses[0][0])
                messagebox.showinfo("Predicted Object", "Lets gooo I AM SMART!!", parent=self.root)
            else:
                messagebox.showinfo("Predicted Object", "OOF, sorry, pls press the class of the actual object :(", parent=self.root)

    def saveModel(self):
        model_weights_path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("HDF5 files", "*.h5")])
        self.model.save_weights(model_weights_path)
        self.model.save_weights(f"QuickDrawData/model_weights.h5")

def main():
    DrawingBoardGraphics()

if __name__ == "__main__":
    main()