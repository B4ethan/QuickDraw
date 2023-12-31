import json
import os
import numpy as np

import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox

import cv2 as cv
import PIL
from PIL import Image, ImageDraw

from tensorflow import keras
from keras import layers, models
from keras.constraints import MaxNorm

class DrawingRecognizer:
    def __init__(self):
        self.class1Counter, self.class2Counter, self.class3Counter = None, None, None
        self.model = None
        self.projectName = None
        self.root = None
        self.tempImage = None
        self.statusLabel = None
        self.canvas = None
        self.draw = None
        self.brush_width = 15

        with open("TestLetsGo\\ObjectsList.json", "r") as data:
            self.dataList = json.load(data)

        self.dataLength = len(self.dataList)

        self.model = self.initCnnModel()
        self.initGui()

    def initCnnModel(self):
        model = keras.Sequential()
        model.add(layers.Conv2D(32, (3, 3), input_shape=(50, 50, 1), padding='same', activation='relu', kernel_constraint=MaxNorm(3)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=MaxNorm(3)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu', kernel_constraint=MaxNorm(3)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.dataLength, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def initGui(self):
        WIDTH = 500
        HEIGHT = 500
        WHITE = (255, 255, 255)

        self.root = tk.Tk()
        self.root.title(f"Drawing Classifier - {self.projectName}")

        self.canvas = tk.Canvas(self.root, width=WIDTH-10, height=HEIGHT-10, bg="white")
        self.canvas.pack(expand= tk.YES, fill= tk.BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.tempImage = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = ImageDraw.Draw(self.tempImage)

        btnFrame = tk.Frame(self.root)
        btnFrame.pack(fill= tk.X, side= tk.BOTTOM)

        btnFrame.columnconfigure(0, weight=1)
        btnFrame.columnconfigure(1, weight=1)
        btnFrame.columnconfigure(2, weight=1)

        saveDrawingBtn = tk.Button(btnFrame, text = "Save Model", command = lambda: self.saveDrawing())
        saveDrawingBtn.grid(row = 0, column= 1, sticky= tk.W + tk.E)

        bminusBtn = tk.Button(btnFrame, text="Brush-", command=self.brushMinus)
        bminusBtn.grid(row=0, column=0, sticky=tk.W + tk.E)

        clearBtn = tk.Button(btnFrame, text="Clear", command=self.clear)
        clearBtn.grid(row=1, column=1, sticky=tk.W + tk.E)

        bplusBtn = tk.Button(btnFrame, text="Brush+", command=self.brushPlus)
        bplusBtn.grid(row=0, column=2, sticky=tk.W + tk.E)

        trainBtn = tk.Button(btnFrame, text="Train Model", command=self.trainModel)
        trainBtn.grid(row=2, column=0, sticky=tk.W + tk.E)

        saveBtn = tk.Button(btnFrame, text="Save Model", command=self.saveModel)
        saveBtn.grid(row=2, column=1, sticky=tk.W + tk.E)

        loadBtn = tk.Button(btnFrame, text="Load Model", command=self.loadModel)
        loadBtn.grid(row=2, column=2, sticky=tk.W + tk.E)

        predictBtn = tk.Button(btnFrame, text="Predict", command=self.predict)
        predictBtn.grid(row=3, column=1, sticky=tk.W + tk.E)

        saveAllBtn = tk.Button(btnFrame, text="Save Everything", command=self.saveAll)
        saveAllBtn.grid(row=3, column=2, sticky=tk.W + tk.E)

        self.statusLabel = tk.Label(btnFrame, text=f"Current Model: {type(self.model).__name__}")
        self.statusLabel.config(font=("Arial", 10))
        self.statusLabel.grid(row=4, column=1, sticky=tk.W + tk.E)

        self.root.protocol("WM_DELETE_WINDOW", self.onClose)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_width)
        self.draw.rectangle([x1, y2, x2 + self.brush_width, y2 + self.brush_width], fill="black", width=self.brush_width)

    def save(self, classNum):
        self.tempImage.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.Resampling.LANCZOS)

        if classNum == 1:
            img.save(f"{self.projectName}/{self.class1}/{self.class1Counter}.png", "PNG")
            self.class1Counter += 1
        elif classNum == 2:
            img.save(f"{self.projectName}/{self.class2}/{self.class2Counter}.png", "PNG")
            self.class2Counter += 1
        elif classNum == 3:
            img.save(f"{self.projectName}/{self.class3}/{self.class3Counter}.png", "PNG")
            self.class3Counter += 1

        self.clear() 

    def saveDrawing(self, objectName = None):
        #popping a question with the name of the object that was drawn, if the class is exists in the data, save the drawing, else, create a new directory with the drawing
        if objectName == None:
            objectName = simpledialog.askstring("Drawing saving", "What is the object in the drawing?", parent=self.msg)

        self.tempImage.save("currentDrawing.png")
        img = PIL.Image.open("currentDrawing.png")
        img.thumbnail((50, 50), PIL.Image.Resampling.LANCZOS)

        with open("TestLetsGo\\ObjectsList.json", "r") as data:
            dataList = json.load(data)

        if objectName in dataList: #if the directory exists
            dataList[objectName] += 1
            img.save(f"{self.projectName}/{objectName}/{dataList[objectName]}.png", "PNG")
        
        else:
            dataList[objectName] = 1
            os.chdir(self.projectName)
            os.mkdir(objectName)
            os.chdir("..")

            img.save(f"{self.projectName}/{objectName}/{dataList[objectName]}.png", "PNG")

        with open("TestLetsGo\\ObjectsList.json", "w") as data:
            json.dump(dataList, data)

        self.clear()
        messagebox.showinfo("Drawing saving", f"Drawing successfully in directory {objectName}", parent = self.root)

    def brushMinus(self):
        if self.brush_width > 1:
            self.brush_width -= 1
    
    def brushPlus(self):
        self.brush_width += 1

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white")

    def saveModel(self):
        # Save model weights to an HDF5 file
        model_weights_path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("HDF5 files", "*.h5")])
        self.model.save_weights(model_weights_path)

        messagebox.showinfo("Drawing Classifier", "Model successfully saved!", parent=self.root)

    def loadModel(self):
        # Load model architecture from a JSON file
        model_architecture_path = filedialog.askopenfilename()
        with open(model_architecture_path, "r") as json_file:
            loaded_model = models.model_from_json(json_file.read())

        # Load model weights from an HDF5 file
        model_weights_path = filedialog.askopenfilename()
        loaded_model.load_weights(model_weights_path)

        self.model = loaded_model

        messagebox.showinfo("Drawing Classifier", "Model successfully loaded!", parent=self.root)

    def saveAll(self):
        # Save the model weights separately
        self.model.save_weights(f"{self.projectName}/{self.projectName}_model_weights.h5")

        messagebox.showinfo("Drawing Classifier", "Project successfully saved!", parent=self.root)

    def onClose(self):
        answer = messagebox.askyesnocancel("Quit?", "Do you want to save your work?", parent=self.root)
        if answer is not None:
            if answer:
                self.saveAll()
                self.root.destroy()
                exit()
            self.root.destroy()
            exit()

    def trainModel(self):
        imgList = []
        objectList = []

        with open("TestLetsGo\\ObjectsList.json", "r") as data:
            objectsInfo = json.load(data)

        labelEncoder = {objectName: label for label, objectName in enumerate(objectsInfo.keys())}
        print(labelEncoder)

        for objectName, objectCounter in objectsInfo.items():
            for drawingNum in range(1, objectCounter + 1):
                imgPath = f"{self.projectName}/{objectName}/{drawingNum}.png"
                img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

                if img is not None:
                    img = cv.resize(img, (50, 50))
                    img = img.reshape((50, 50, 1))
                    imgList.append(img)
                    objectList.append(labelEncoder[objectName])
                else:
                    print(f"Error loading image: {imgPath}")

        imgArr = np.array(imgList)
        objectArr = np.array(objectList)

        print(np.unique(objectArr))

        imgArr = imgArr / 255.0

        self.model.fit(imgArr, objectArr, epochs=20)
        
        messagebox.showinfo("Drawing Classifier", "Model successfully trained!", parent=self.root)

       
    def predict(self):

        with open("TestLetsGo\\ObjectsList.json", "r") as ObjectList:
            objectNamesList = json.load(ObjectList)

        # Save the current drawing to a temporary file
        self.tempImage.save("temp.png")
        img = cv.imread("temp.png", cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (50, 50))
        img = img.reshape((1, 50, 50, 1))

        # Normalize pixel values to be between 0 and 1
        img = img / 255.0

        # Make a prediction using the CNN model
        prediction = self.model.predict(img)[0]

        objectPredict = {objectName: prediction[idx] * 100 for idx, objectName in enumerate(objectNamesList.keys())}

        sortedDrawings = sorted(objectPredict.keys(), key=lambda x: objectPredict[x], reverse=True)
        
        threshold = 0.5
        topThreeGuesses = [(class_name, objectPredict[class_name]) for class_name in sortedDrawings[:3] if objectPredict[class_name] >= threshold]



        message = "I think the drawing is:\n"
        for i, (predictedClass, confidence) in enumerate(topThreeGuesses):
            message += f"{i+1}. {predictedClass} with {confidence:.2f}% confidence\n"

        isCorrect = messagebox.askyesno("Predicted Object", f"{message}Am I right?", parent=self.root)

        if isCorrect is not None:
            if isCorrect:
                self.saveDrawing(topThreeGuesses[0][0])
                messagebox.showinfo("Predicted Object", "Lets gooo I AM SMART!!", parent=self.root)
            else:
                messagebox.showinfo("Predicted Object", "OOF, sorry, pls press the class of the actual object :(", parent=self.root)
def main():
    DrawingRecognizer()

if __name__ == "__main__":
    main()

