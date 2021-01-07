import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from keras.models import load_model
from tkinter import *
from tkinter import filedialog
import PIL.Image
import PIL.ImageTk
loaded_model = load_model("pneumonia_detection_model.h5")
root = Tk()
root.title("Pneumonia Detection")
root.state('zoomed')
root.configure(bg='#D3D3D3')
root.resizable(width = True, height = True) 
value = StringVar()
panel = Label(root)

def detect(filename):
    img = plt.imread(filename)
    img = cv2.resize(img, (150,150))
    img = np.dstack([img, img, img])
    img = img.astype('float32')/255
    img = np.expand_dims(img, axis=0)
    pred = np.round(loaded_model.predict(img)[0])
    if(pred == 1):
        value.set("PNEUMONIA!")
    else:
        value.set("NORMAL")

def ClickAction(event=None):
    filename = filedialog.askopenfilename()
    img = PIL.Image.open(filename)
    img = img.resize((250,250))
    img = PIL.ImageTk.PhotoImage(img)
    global panel
    panel = Label(root, image = img)
    panel.image = img
    panel = panel.place(relx=0.435,rely=0.3)
    detect(filename)

button = Button(root, text='CHOOSE FILE', font=(None, 18), activeforeground='red', bd=20, bg='cyan', relief=RAISED, height=3, width=20, command=ClickAction)
button = button.place(relx=0.40, rely=0.05)
result = Label(root, textvariable=value, font=(None, 20))
result = result.place(relx=0.465,rely=0.7)
root.mainloop()