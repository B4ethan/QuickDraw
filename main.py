import json
import os
import numpy as np

import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox

import cv2 as cv
import PIL
from PIL import Image, ImageDraw

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

class DrawingRecognizer:
    def __init__(self):
        self.class1, self.class2, self.class3 = None, None, None
        self.class1Counter, self.class2Counter, self.class3Counter = None, None, None
        self.model = None
        self.projectName = None
        self.root = None
        self.image1 = None
        self.status_label = None
        self.canvas = None
        self.draw = None
        self.brush_width = 15

        self.classesPrompt()
        self.initCnnModel()
        self.initGui()

    def classesPrompt(self):
        msg = tk.Tk()
        msg.withdraw()

        self.projectName = simpledialog.askstring("Project Name", "Enter project name", parent=msg)

        project_data_path = f"{self.projectName}/{self.projectName}.json"

        if os.path.exists(project_data_path):
            with open(project_data_path, "r") as file:
                data = json.load(file)

            self.class1 = data['c1']
            self.class2 = data['c2']
            self.class3 = data['c3']
            self.class1Counter = len(os.listdir(f"{self.projectName}/{self.class1}")) + 1
            self.class2Counter = len(os.listdir(f"{self.projectName}/{self.class2}")) + 1
            self.class3Counter = len(os.listdir(f"{self.projectName}/{self.class3}")) + 1

            # Rebuild the model architecture
            self.model = self.initCnnModel()

            # Load the model weights
            self.model.load_weights(f"{self.projectName}/{self.projectName}_model_weights.h5")

        else:
            self.class1 = simpledialog.askstring("Object 1", "What is the first object?", parent=msg)
            self.class2 = simpledialog.askstring("Object 2", "What is the second object?", parent=msg)
            self.class3 = simpledialog.askstring("Object 3", "What is the third object?", parent=msg)

            self.class1Counter = 1
            self.class2Counter = 1
            self.class3Counter = 1

            self.model = self.initCnnModel()
            os.mkdir(self.projectName)
            os.chdir(self.projectName)
            os.mkdir(self.class1)
            os.mkdir(self.class2)
            os.mkdir(self.class3)
            os.chdir("..")

    def initCnnModel(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(3, activation='softmax'))  # Adjust output size based on the number of classes

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

        self.image1 = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = ImageDraw.Draw(self.image1)

        btnFrame = tk.Frame(self.root)
        btnFrame.pack(fill= tk.X, side= tk.BOTTOM)

        btnFrame.columnconfigure(0, weight=1)
        btnFrame.columnconfigure(1, weight=1)
        btnFrame.columnconfigure(2, weight=1)

        class1Btn = tk.Button(btnFrame, text=self.class1, command=lambda: self.save(1))
        class1Btn.grid(row=0, column=0, sticky=tk.W + tk.E)

        class2Btn = tk.Button(btnFrame, text=self.class2, command=lambda: self.save(2))
        class2Btn.grid(row=0, column=1, sticky=tk.W + tk.E)

        class3Btn = tk.Button(btnFrame, text=self.class3, command=lambda: self.save(3))
        class3Btn.grid(row=0, column=2, sticky=tk.W + tk.E)

        bminusBtn = tk.Button(btnFrame, text="Brush-", command=self.brushMinus)
        bminusBtn.grid(row=1, column=0, sticky=tk.W + tk.E)

        clearBtn = tk.Button(btnFrame, text="Clear", command=self.clear)
        clearBtn.grid(row=1, column=1, sticky=tk.W + tk.E)

        bplusBtn = tk.Button(btnFrame, text="Brush+", command=self.brushPlus)
        bplusBtn.grid(row=1, column=2, sticky=tk.W + tk.E)

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

        self.status_label = tk.Label(btnFrame, text=f"Current Model: {type(self.model).__name__}")
        self.status_label.config(font=("Arial", 10))
        self.status_label.grid(row=4, column=1, sticky=tk.W + tk.E)

        self.root.protocol("WM_DELETE_WINDOW", self.onClose)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_width)
        self.draw.rectangle([x1, y2, x2 + self.brush_width, y2 + self.brush_width], fill="black", width=self.brush_width)

    def save(self, classNum):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.ANTIALIAS)

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

    def brushMinus(self):
        if self.brush_width > 1:
            self.brush_width -= 1
    
    def brushPlus(self):
        self.brush_width += 1

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white")

    def saveModel(self):
        # Save model architecture to a JSON file
        model_architecture_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        with open(model_architecture_path, "w") as json_file:
            json_file.write(self.model.to_json())

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
        data = {"c1": self.class1, "c2": self.class2, "c3": self.class3, "c1c": self.class1Counter,
                "c2c": self.class2Counter, "c3c": self.class3Counter, "pname": self.projectName}
        
        # Save project data to a JSON file
        with open(f"{self.projectName}/{self.projectName}.json", "w") as json_file:
            json.dump(data, json_file)

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
        img_list = []
        class_list = []

        for x in range(1, self.class1Counter):
            imgPath = f"{self.projectName}/{self.class1}/{x}.png"
            img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

            if img is not None:  # Check if the image is successfully loaded
                img = cv.resize(img, (50, 50))
                img = img.reshape((50, 50, 1))
                img_list.append(img)
                class_list.append(0)  # Assuming class1 is the first class, use 0 as its label
            else:
                print(f"Error loading image: {imgPath}")

        for x in range(1, self.class2Counter):
            imgPath = f"{self.projectName}/{self.class2}/{x}.png"
            img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

            if img is not None:  # Check if the image is successfully loaded
                img = cv.resize(img, (50, 50))
                img = img.reshape((50, 50, 1))
                img_list.append(img)
                class_list.append(1)  # Assuming class2 is the second class, use 1 as its label
            else:
                print(f"Error loading image: {imgPath}")

        for x in range(1, self.class3Counter):
            imgPath = f"{self.projectName}/{self.class3}/{x}.png"
            img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

            if img is not None:  # Check if the image is successfully loaded
                img = cv.resize(img, (50, 50))
                img = img.reshape((50, 50, 1))
                img_list.append(img)
                class_list.append(2)  # Assuming class3 is the third class, use 2 as its label
            else:
                print(f"Error loading image: {imgPath}")

        img_array = np.array(img_list)
        class_array = np.array(class_list)

        # Normalize pixel values to be between 0 and 1
        img_array = img_array / 255.0

        # Train the CNN model
        self.model.fit(img_array, class_array, epochs=10)  # Adjust epochs as needed

        # Show a message box indicating successful training
        messagebox.showinfo("Drawing Classifier", "Model successfully trained!", parent=self.root)

    def predict(self):
        # Save the current drawing to a temporary file
        self.image1.save("temp.png")
        img = cv.imread("temp.png", cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (50, 50))
        img = img.reshape((1, 50, 50, 1))

        # Normalize pixel values to be between 0 and 1
        img = img / 255.0

        # Make a prediction using the CNN model
        prediction = self.model.predict(img)

        # Get the predicted class index
        predictedClassIndex = np.argmax(prediction)

        # Map the class index to the actual class label
        classNames = [self.class1, self.class2, self.class3]
        predictedClass = classNames[predictedClassIndex]

        # Show a message box with the predicted class
        isCorrect = messagebox.askyesno("Predicted Object", f"I think the drawing is {predictedClass}\n am i right?", parent=self.root)

        if isCorrect is not None:
            if isCorrect:
                self.save(predictedClassIndex + 1)
                messagebox.showinfo("Predicted Object", "Lets gooo I AM SMART!!", parent = self.root)
            else:
                messagebox.showinfo("Predicted Object", "OOF, sorry, pls press the class of the actual object :(", parent = self.root)

def main():
    DrawingRecognizer()

if __name__ == "__main__":
    main()

