import pickle
import os.path

#graphic libraray
from tkinter import *
from tkinter import simpledialog, filedialog
from tkinter import messagebox

import numpy as np
import cv2 as cv
import PIL
import PIL.Image
import PIL.ImageDraw

#Ai models
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

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
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = PIL.ImageDraw.Draw(self.image1)

        btnFrame = Frame(self.root)
        btnFrame.pack(fill = X, side = BOTTOM)
        
        btnFrame.columnconfigure(0, weight = 1)
        btnFrame.columnconfigure(2, weight = 1)
        btnFrame.columnconfigure(2, weight = 1)

        class1Btn = Button(btnFrame, text = self.class1, command= lambda: self.save(1))
        class1Btn.grid(row = 0, column= 0, sticky= W + E)
        
        class2Btn = Button(btnFrame, text = self.class2, command= lambda: self.save(2))
        class2Btn.grid(row = 0, column= 1, sticky= W + E)

        class3Btn = Button(btnFrame, text = self.class3, command= lambda: self.save(3))
        class3Btn.grid(row = 0, column= 2, sticky= W + E)

        clearBtn = Button(btnFrame, text = "Clear", command= self.clear)
        clearBtn.grid(row=1, column=1, sticky= W + E)

        trainBtn = Button(btnFrame, text = "Train Model", command=self.trainModel)
        trainBtn.grid(row=2, column=0, sticky= W + E)

        saveBtn = Button(btnFrame, text = "Save", command=self.saveModel)
        saveBtn.grid(row=2, column=1, sticky= W + E)

        loadBtn = Button(btnFrame, text = "Load Model", command=self.loadModel)
        loadBtn.grid(row=2, column=2, sticky= W + E)

        changeModelBtn = Button(btnFrame, text= "Change Model", command=self.changeModel)
        changeModelBtn.grid(row=3, column=0, sticky= W + E)

        predictBtn = Button(btnFrame, text = "Predict Model", command=self.predict)
        predictBtn.grid(row=3, column=1, sticky= W + E)

        loadBtn = Button(btnFrame, text = "Load Model", command=self.loadModel)
        loadBtn.grid(row=3, column=2, sticky= W + E)

        saveAllBtn = Button(btnFrame, text= "Save all", command=self.saveAll)
        saveAllBtn.grid(row=3, column=2, sticky= W + E)

        self.statusLabel = Label(btnFrame, text=f"Current Model: {type(self.clf).__name__}")
        self.statusLabel.config(font=("David", 10))
        self.statusLabel.grid(row=4, column=1, sticky= W + E)

        self.root.protocol("WM_DELETE_WINDOW", self.onClose())
        self.root.attributes("-topmost", True)
        self.root.mainloop()


    def paint(self, event):
        x1, y1 = (event.x -1), (event.y -1)
        x2, y2 = (event.x +1), (event.y +1)

        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brushSize)
        self.draw.rectangle([x1, y2, x2 + self.brushSize, y2 + self.brushSize], fill="black", width=self.brushSize)

    def save(self, classNumber):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.Resampling.LANCZOS)

        if classNumber == 1: 
            img.save(f"{self.projectName}/{self.class1}/{self.class1Counter}.png", "PNG")
            self.class1Counter += 1

        elif classNumber == 2:
            img.save(f"{self.projectName}/{self.class2}/{self.class2Counter}.png", "PNG")
            self.class2Counter += 1

        else:
            img.save(f"{self.projectName}/{self.class3}/{self.class3Counter}.png", "PNG")
            self.class3Counter += 1

        self.clear()

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white")

    def changeModel(self):
        if isinstance(self.clf, LinearSVC):
            self.clf = KNeighborsClassifier()

        elif isinstance(self.clf, KNeighborsClassifier):
            self.clf = LinearSVC()

        self.statusLabel.config(text=f"Current Model: {type(self.clf).__name__}")

    def trainModel(self):
        imgList = np.array([])
        objectList = np.array([])

        for i in range(1, self.class1Counter):
            img = cv.imread(f"{self.projectName}/{self.class1}/{i}.png")[:, :, 0]
            img = cv.resize(img, 2500)
            imgList = np.append(imgList, [img])
            objectList = np.append(objectList, 1)

        for i in range(1, self.class2Counter):
            img = cv.imread(f"{self.projectName}/{self.class2}/{i}.png")[:, :, 0]
            img = cv.resize(img, 2500)
            imgList = np.append(imgList, [img])
            objectList = np.append(objectList, 2)

        for i in range(1, self.class3Counter):
            img = cv.imread(f"{self.projectName}/{self.class3}/{i}.png")[:, :, 0]
            img = cv.resize(img, 2500)
            imgList = np.append(imgList, [img])
            objectList = np.append(objectList, 3)

        imgList = imgList.reshape(self.class1Counter - 1 + self.class2Counter -1 + self.class3Counter - 1, 2500)

        self.clf.fit(imgList, objectList)
        messagebox.showinfo("Drawing classfier", "Model is traind!", parent=self.root)

    def predict(self):
        self.image1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.Resampling.LANCZOS)
        img.save("predictedObject", "PNG")

        img = cv.imread("predictedObject.png")[:, :, 0]
        img = img.reshape(2500)
        prediction = self.clf.predict([img])

        if prediction[0] == 1:
            messagebox.showinfo("Drawing classfier", f"I think that the drawing is a {self.class1}", parent=self.root)

        elif prediction[0] == 2:
            messagebox.showinfo("Drawing classfier", f"I think that the drawing is a {self.class2}", parent=self.root)

        elif prediction[0] == 3:
            messagebox.showinfo("Drawing classfier", f"I think that the drawing is a {self.class3}", parent=self.root)

    def saveModel(self):
        filePath = filedialog.asksaveasfilename(defaultextension="pickle")
        with open(filePath, "wb") as f:
            pickle.dump(self.clf, f)

        messagebox.showinfo("Drawing classifeir", "Model saved!", parent= self.root)

    def loadModel(self):
        filePath = filedialog.askopenfilename()
        with open(filePath, "rb") as f:
            self.clf = pickle.load(f)

        messagebox.showinfo("Drawing classifeir", "Model loaded!", parent= self.root)

    def saveAll(self):
        data = {
            "c1" : self.class1, 
            "c2" : self.class2, 
            "c3" : self.class3, 
            "c1c" : self.class1Counter,
            "c2c" : self.class2Counter,
            "c3c" : self.class3Counter,
            "clf" : self.clf,
            "pname" : self.projectName}
        
        with open(f"{self.projectName}/{self.projectName}_data.pickle", "rb") as f:
            pickle.dump(data, f)

            messagebox.showinfo("Drawing classifeir", "project Save!", parent= self.root)

    def onClose(self):
        answer = messagebox.askyesnocancel("Quit?", "Do you want to save your work?", parent=self.root)

        if answer is not None:
            if answer:
                self.saveAll()

            self.root.destroy()
            exit()

def main():
    DrawingRecognizer()

if __name__ == "__main__":
    main()