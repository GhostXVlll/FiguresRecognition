from tkinter import *
from PIL import Image, ImageDraw
import Lab3


def draw(event):
    x1, y1 = (event.x - brushSize), (event.y-brushSize)
    x2, y2 = (event.x - brushSize), (event.y-brushSize)
    canvas.create_oval(x1, y1, x2, y2, fill=color, width=0)
    draw_img.ellipse((x1, y1, x2, y2), fill=color, width=0)


def clear_canvas():
    canvas.delete('all')
    canvas['bg'] = 'black'
    draw_img.rectangle((0, 0, 28, 28), width=0, fill='black')


def predict():
    filename = 'image.png'
    image1.save(filename)

    Lab3.pred()


x = 0
y = 0

root = Tk()
root.title('Lab3 AIS NeuralNetwork')
root.geometry("500x500")  # Window scale
root.resizable(False, False)

brushSize = 10
color = "white"

root.columnconfigure(6, weight=1)
root.rowconfigure(2, weight=1)

canvas = Canvas(root, bg="black")
canvas.grid(row=2, column=0, columnspan=7, padx=5, pady=5)

canvas.bind('<B1-Motion>', draw)

image1 = Image.new('RGB', (28, 28), 'white')  # Image scale
draw_img = ImageDraw.Draw(image1)

Label(root, text='Draw the figure from \"0\" to \"9\" \n').grid(row=1, column=0, padx=6)
Button(root, text='Clear: ', width=10, command=clear_canvas).grid(row=1, column=1)
Button(root, text='Predict figure: ', width=10, command=predict).grid(row=1, column=2)

root.mainloop()
