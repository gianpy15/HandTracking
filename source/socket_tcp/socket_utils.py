import socket

host = 'localhost'
port = 8343
packet_size = 1024


def connect_to_camera():
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
    frame = tcp_socket.recv(packet_size)

    while len(frame) < size:
        frame += tcp_socket.recv(packet_size)

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
    tcp_socket = connect_to_camera()

    while 1:
        rgb_frame = get_rgb_frame(tcp_socket)
        # use rgb_frame
        print(rgb_frame)

        z16_frame = get_z16_frame(tcp_socket)
        # use z16_frame
        print(z16_frame)

    disconnect_camera(tcp_socket)
