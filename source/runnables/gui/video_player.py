import sys                                                                                                                                                               
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..")))

from tkinter import *
from library.gui.player_thread import PlayerThread
from library.gui.model_drawer import *
from data.datasets.framedata_management.video_loader import load_labeled_video
from data.datasets.framedata_management.camera_data_conversion import *
import threading
import skvideo.io as skio
from data.naming import *
from tkinter.simpledialog import askstring
import os.path as path

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

    if vidname is not None and vidname != "":

        # Load frames and labels
        # vidname = "snap"
        split = path.splitext(vidname)
        isdepth = False
        try:
            if split[1] == ".z16":
                frames, labels, indexes = load_labeled_video(split[0], getdepth=True, gapflags=True)
                isdepth = True
            elif split[1] == ".rgb":
                frames, labels, indexes = load_labeled_video(split[0], gapflags=True)
            else:
                frames, labels, indexes = load_labeled_video(vidname, gapflags=True)
        except FileNotFoundError:
            isdepth = False
            try:
                if split[1] == ".rgb":
                    videopath = path.join(resources_path("rawcam"), split[0])
                elif split[1] == ".z16":
                    videopath = path.join(resources_path("rawcam"), split[0])
                    isdepth = True
                else:
                    videopath = path.join(resources_path("rawcam"), vidname)

                if isdepth:
                    frames = read_frame_data(**default_read_z16_args(framesdir=videopath))
                else:
                    frames = read_frame_data(**default_read_rgb_args(framesdir=videopath))
                labels = None
                indexes = None
            except FileNotFoundError:
                isdepth = False
                try:
                    videopath = path.join(resources_path("vids"), vidname)
                    frames = skio.vread(videopath)
                    labels = None
                    indexes = None
                except Exception as e:
                    frames = None
                    print("Unable to load the video %s, file not found." % vidname)
                    exit(-1)

        if frames[0] is None:
            print("Video contains Nones, thus it is invalid. Have you tried to load depth from a video without depth?")
            exit(-1)

        if isdepth:
            frames = np.repeat(enhance_depth_vid(frames), 3, axis=3)

        # SAMPLE CODE TO MESH BLUESCREEN VIDEO WITH BACKGROUND VIDEO
        # bkg, _ = load_labeled_video("hands_A", False, False, False)
        # bkg = bkg[0:len(frames)]

        # from framedata_management.bluescreen import make_bluescreen_filter
        # import tqdm
        # for idx in tqdm.tqdm(range(len(frames))):
        #     filter = make_bluescreen_filter(frames[idx])
        #     frames[idx, filter] = bkg[idx, filter]

        image_width = np.shape(frames)[2]
        image_height = np.shape(frames)[1]

        topframe = Frame(root, height=image_height, width=cmd_width + image_width)
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

        # Keep button
        keep_button = Button(cmd, text="Keep")
        keep_button.pack(fill=BOTH)

        # Keep All button
        keep_all_button = Button(cmd, text="Keep All")
        keep_all_button.pack(fill=BOTH)

        # Next button
        next_button = Button(cmd, text="Next")
        next_button.pack(fill=BOTH)

        # Next Fixed button
        next_fixed_button = Button(cmd, text="Next fixed")
        next_fixed_button.pack(fill=BOTH)

        # Prev button
        prev_button = Button(cmd, text="Previous")
        prev_button.pack(fill=BOTH)

        # Prev Fixed button
        prev_fixed_button = Button(cmd, text="Previous fixed")
        prev_fixed_button.pack(fill=BOTH)

        # Interpolate button
        interp_button = Button(cmd, text="Interpolate fixed")
        interp_button.pack(fill=BOTH)

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
            discard=discard,
        )

        # keypress binding 
        canvas.bind('p', lambda e: player.play())
        canvas.bind('o', lambda e: player.pause())
        canvas.bind('k', lambda e: player.keepthis())
        canvas.bind('l', lambda e: player.keepall())
        canvas.bind('d', lambda e: player.set_changes())
        canvas.bind('s', lambda e: player.print_changes(vidname))
        canvas.bind('<Up>', lambda e: player.set_current_frame(player.current_frame+1))
        canvas.bind('<Right>', lambda e: player.next_fixed_frame())
        canvas.bind('<Down>', lambda e: player.set_current_frame(player.current_frame-1))
        canvas.bind('<Left>', lambda e: player.next_fixed_frame(jumps=-1))
        canvas.bind('a', lambda e: player.reinterpolate())

        canvas.focus_set()

        pause_button.bind('<Button-1>', lambda e: player.pause())
        play_button.bind('<Button-1>', lambda e: player.play())
        keep_button.bind('<Button-1>', lambda e: player.keepthis())
        keep_all_button.bind('<Button-1>', lambda e: player.keepall())
        root.after(50, player.nextframe, root)

        slider = Scale(root,
                       from_=-2., to=3., resolution=0.05,
                       orient=HORIZONTAL, tickinterval=1.0,
                       command=lambda v: player.set_speed_mult(v))
        slider.set(1)
        slider.pack(side=BOTTOM, fill=BOTH)
        frameslider = Scale(root,
                            from_=0, to=np.shape(frames)[0] - 1, resolution=1,
                            orient=HORIZONTAL, tickinterval=20.0,
                            command=lambda v: player.set_current_frame(int(v)))
        frameslider.set(0)
        player.frameslider = frameslider
        frameslider.pack(side=BOTTOM, fill=BOTH)

        discard_button.bind('<Button-1>', lambda e: player.set_changes())

        save_button.bind('<Button-1>', lambda e: player.print_changes(vidname))

        next_button.bind('<Button-1>', lambda e: player.set_current_frame(player.current_frame+1))

        next_fixed_button.bind('<Button-1>', lambda e: player.next_fixed_frame())

        prev_button.bind('<Button-1>', lambda e: player.set_current_frame(player.current_frame-1))

        prev_fixed_button.bind('<Button-1>', lambda e: player.next_fixed_frame(jumps=-1))

        interp_button.bind('<Button-1>', lambda e: player.reinterpolate())

        root.mainloop()
