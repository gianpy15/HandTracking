import socket

PACKET_SIZE = 1024


class Z300Streamer:
    """
    This class manages the data stream from the Z300 camera.
    This is supposed to stream from a dedicated thread.
    Data is streamed through an observer pattern.
    """
    def __init__(self, host='localhost', port=8343):
        self.host = host
        self.port = port
        self.active = False
        self.rgblisteners = []
        self.z16listeners = []
        self.fulllisteners = []

    def stream(self):
        """
        Connect to the camera and start streaming the received data.
        Each time a frame is completed, the respective listener functions are called.
        This is an infinite loop that ends as soon as the disconnect method is called
        from outside or by any listener function.
        """
        connection = connect_to_camera(host=self.host,
                                       port=self.port)
        self.active = True
        while self.active:
            rgb = get_rgb_frame(connection)
            for lst in self.rgblisteners:
                lst(rgb)
            z16 = get_z16_frame(connection)
            for lst in self.z16listeners:
                lst(z16)
            for lst in self.fulllisteners:
                lst(rgb, z16)

        disconnect_camera(connection)

    def disconnect(self):
        """
        End the main streaming loop.
        """
        self.active = False

    def add_rgb_listener(self, func):
        """
        Register a listener to be called with rgb frames only.
        :param func: a function taking one argument, the rgb frame.
        """
        self.rgblisteners.append(func)

    def add_z16_listener(self, func):
        """
        Register a listener to be called with depth frames only.
        :param func: a function taking one argument, the z16 frame.
        """
        self.z16listeners.append(func)

    def add_full_listener(self, func):
        """
        Register a listener to be called with corresponding rgb and depth frames.
        :param func: a function taking two arguments, rgb and z16 frames respectively.
        """
        self.fulllisteners.append(func)


def connect_to_camera(host='localhost', port=8343):
    """
    Connects to the RealSense camera
    :return: the socket over which the streaming is transmitted
    """
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect((host, port))
    return tcp_socket


def __get_frame(tcp_socket, size):
    """
    Reads size bytes
    :param tcp_socket: the socket to be listened to
    :param size: the size of the entire frame
    :return: the complete frame
    """
    frame = tcp_socket.recv(PACKET_SIZE)

    while len(frame) < size:
        frame += tcp_socket.recv(PACKET_SIZE)

    return frame


def get_rgb_frame(tcp_socket):
    """
    Reads one RGB frame
    :param tcp_socket: the socket to be listened to
    :return: the complete RGB frame
    """
    return __get_frame(tcp_socket, 640 * 480 * 3)


def get_z16_frame(tcp_socket):
    """
    Reads one Z16 frame
    :param tcp_socket: the socket to be listened to
    :return: the complete Z16 frame
    """
    return __get_frame(tcp_socket, 640 * 480 * 2)


def disconnect_camera(tcp_socket):
    """
    Disconnects the camera, by closing the socket opened to handle it
    :param tcp_socket: the socket used for connecting to the camera
    :return: True if the connection was closed successfully
    """
    tcp_socket.shutdown()
    tcp_socket.close()
    return True


if __name__ == "__main__":
    from gui.frame_displayer import FrameDisplayer
    import tkinter as tk
    import numpy as np
    import threading as tr

    root = tk.Tk()
    height, width = 480, 640

    canvas = tk.Canvas(root)
    canvas.pack()

    fd = FrameDisplayer(canvas, "RGB")

    streamer = Z300Streamer()

    streamer.add_rgb_listener(lambda f: fd.update_frame(np.array(f,
                                                                 dtype=np.uint8)))

    tr.Thread(target=Z300Streamer.stream).start()

    root.mainloop()

