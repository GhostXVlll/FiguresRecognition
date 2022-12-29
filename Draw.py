import io
from tkinter import *
from PIL import Image, ImageDraw
import Lab3


def draw(event):
    x1, y1 = (event.x - brushSize), (event.y - brushSize)
    x2, y2 = (event.x + brushSize), (event.y + brushSize)
    canvas.create_oval(x1, y1, x2, y2, fill=color, width=0)
    draw_img.ellipse((x1, y1, x2, y2), fill=color, width=0)


def clear_canvas():
    canvas.delete('all')
    canvas['bg'] = 'white'
    draw_img.rectangle((0, 0, 28, 28), width=0, fill='white')


def predict():
    # filename = 'image.png'
    # image1.save(filename)

    size = 28, 28
    ps = canvas.postscript(colormode="color")
    image = Image.open(io.BytesIO(ps.encode('utf-8')))
    image = image.resize(size)
    image.save("image.png", 'png')

    Lab3.main()


x = 0
y = 0

root = Tk()
root.title('Lab3 AIS NeuralNetwork')
root.geometry("500x500")  # Window scale
root.resizable(False, False)

brushSize = 10
color = "black"

root.columnconfigure(6, weight=1)
root.rowconfigure(2, weight=1)

canvas = Canvas(root, bg="white")
canvas.grid(row=2, column=0, columnspan=7, padx=5, pady=5)

canvas.bind('<B1-Motion>', draw)

image1 = Image.new('1', (28, 28), "white")  # Image scale
draw_img = ImageDraw.Draw(image1)

Label(root, text='Draw the figure from \"0\" to \"9\" \n').grid(row=0, column=0, padx=6)
Button(root, text='Clear: ', width=10, command=clear_canvas).grid(row=1, column=1)
Button(root, text='Recognize figure: ', width=15, command=predict).grid(row=1, column=2)
Button(root, text='Quit', width=10, command=root.quit).grid(row=1, column=4)


root.mainloop()
