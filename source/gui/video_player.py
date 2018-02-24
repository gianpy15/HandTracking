# TODO
# Implement a video player to merge frame data with label data of a certain video
# Suggested classes to use:
# - in gui.model_drawer.py: ModelDrawer
#   instanciate:        md = ModelDrawer()
#   set full canvas:    md.set_target_area(canvas)
#   when needed draw:   md.set_joints(joints)
#   --> see the doc of the class and the two methods
# - in image_loader.hand_io.py: load()
#   --> see the doc of the function
#
# about reading video data and interpolating missing labels
# a standalone module may be done
# because it is needed also for training
# - in hand_data_management.video_loader: function load_labeled_video()
#   --> see the doc of the function
from tkinter import *
from gui.player_thread import PlayerThread
from multiprocessing import *
from gui.model_drawer import *
from geometry.formatting import hand_format
from hand_data_management.video_loader import load_labeled_video
from PIL import Image, ImageTk
import numpy as np

# global variables
play = True
delay = 50000
play_flag = Event()

def play(event):
    play_flag.set()

def pause(event):
    play_flag.clear()

# Dimension
image_height=336#480
image_width=596#640
cmd_width=90
cmd_height=image_height

# Load frames and labels
frames, labels = load_labeled_video("snap") # TODO add choice of video

# Root window of GUI
root = Tk()
root.title("Player")

# Frame for buttons
cmd = Frame(root, height=cmd_height, width=cmd_width)
cmd.pack(side=RIGHT)

# Canvas for frames and labels
canvas = Canvas(root, width=image_width, height=image_height)
canvas.pack(side=LEFT)

# Play button
play_button = Button(cmd, text="Play")
play_button.pack(fill=BOTH)
play_button.bind('<Button-1>', play)

# Pause button
pause_button = Button(cmd, text="Pause")
pause_button.pack(fill=BOTH)
pause_button.bind('<Button-1>', pause)

# Build the model drawer
md = ModelDrawer()
md.set_target_area(canvas)

#if np.array(frames).dtype in [np.float16, np.float32, np.float64, np.float128]:
#    frames_list = np.array(frames * 255, dtype=np.int8)
#elif np.array(frames).dtype.itemsize != 1:
#    frames_list = np.array(frames, dtype=np.int8)
#else:
#    frames_list = frames
#
#play_flag.set()
#for i in range(len(frames_list)):
#    play_flag.wait()
#    image = Image.fromarray(frames_list[i], mode="RGB")
#    photo_image = ImageTk.PhotoImage(image)
#    img = canvas.create_image(image_width/2, image_height/2, image=photo_image)
#    md.set_joints(hand_format(labels[i]))
#    n = delay
#    while n > 0:
#        n = n - 1
#    canvas.update()
#    canvas.delete("all")

# Build and run the thread
player = PlayerThread(frames=frames, labels=labels, canvas=canvas, md=md, flag=play_flag)
player.start()

root.mainloop()