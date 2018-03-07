from tkinter import *
from gui.player_thread import PlayerThread
from gui.model_drawer import *
from hand_data_management.video_loader import load_labeled_video
from hand_data_management.camera_data_conversion import read_frame_data, default_read_rgb_args
import threading
import skvideo.io as skio
from image_loader.hand_io import pm
from tkinter.simpledialog import askstring
from os.path import join

if __name__ == '__main__':
    # global variables
    play = True
    delay = 50000

    play = threading.Condition()

    # Dimension
    image_height = 336  # 480
    image_width = 596  # 640
    cmd_width = 90
    cmd_height = image_height

    # Root window of GUI
    root = Tk()
    root.title("Player")

    # Choose video name
    vidname = askstring(title="Type video's name", parent=root, prompt="Video name:")

    if vidname != None and vidname != "":

        # Load frames and labels
        # vidname = "snap"
        try:
            frames, labels, indexes = load_labeled_video(vidname, gapflags=True)
        except FileNotFoundError:
            try:
                videopath = join(pm.resources_path("rawcam"), vidname)
                frames = read_frame_data(**default_read_rgb_args(framesdir=videopath))
                labels = None
                indexes = None
            except FileNotFoundError:
                try:
                    videopath = join(pm.resources_path("vids"), vidname+".mp4")
                    frames = skio.vread(videopath)
                    labels = None
                    indexes = None
                except Exception as e:
                    print("Unable to load the video %s, file not found." % vidname)
                    exit(-1)

        image_width = np.shape(frames)[2]
        image_height = np.shape(frames)[1]

        topframe = Frame(root, height=image_height, width=cmd_width+image_width)
        topframe.pack(side=TOP)

        # Frame for buttons
        cmd = Frame(topframe, height=cmd_height, width=cmd_width)
        cmd.pack(side=RIGHT)

        # Canvas for frames and labels
        canvas = Canvas(topframe, width=image_width, height=image_height)
        canvas.pack(side=LEFT)

        # Play button
        play_button = Button(cmd, text="Play")
        play_button.pack(fill=BOTH)

        # Pause button
        pause_button = Button(cmd, text="Pause")
        pause_button.pack(fill=BOTH)

        # Labeled/unlabeled message field
        msg = StringVar()
        frame_status = Label(root, textvariable=msg)
        frame_status.pack()

        # Discard commands
        discard_button = Button(cmd, text="Discard")
        discard_button.pack(fill=BOTH)
        discard = StringVar()
        discard_label = Label(root, textvariable=discard)
        discard_label.pack()

        # Save button
        save_button = Button(cmd, text="Save")
        save_button.pack(fill=BOTH)

        # Build the model drawer
        md = ModelDrawer()
        md.set_target_area(canvas)

        # Build and run the thread
        player = PlayerThread(
            frames=frames,
            labels=labels,
            canvas=canvas,
            modeldrawer=md,
            status=msg,
            indexes=indexes,
            discard=discard
        )
        pause_button.bind('<Button-1>', lambda e: player.pause())
        play_button.bind('<Button-1>', lambda e: player.play())
        root.after(50, player.nextframe, root)

        slider = Scale(root,
                       from_=-2., to=3., resolution=0.05,
                       orient=HORIZONTAL, tickinterval=1.0,
                       command=lambda v: player.set_speed_mult(v))
        slider.set(1)
        slider.pack(side=BOTTOM, fill=BOTH)

        discard_button.bind('<Button-1>', lambda e: player.set_changes())

        save_button.bind('<Button-1>', lambda e: player.print_changes(vidname))

        root.mainloop()
