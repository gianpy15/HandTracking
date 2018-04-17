import telepot
import datetime
import socket
import numpy as np
import io
from PIL import Image

BOT_TOKEN = "561223507:AAGvadvBfQcRb3hhTXQN1FN7c2xtn6B9vm0"
CHAT_ID = -307476339


def send_message(message="", disable_notification=False):
    bot = telepot.Bot(BOT_TOKEN)
    bot.sendMessage(CHAT_ID, message, disable_notification=disable_notification)


def notify_training_starting(model_name=None, **kwargs):
    """
    Function for notify that the training is starting
    :param model_name: Is the name of the model that will be trained (optional)
    :param kwargs: If you want to send some util information you can put them here
                   as key=value
    :return: None, it will just send the notify
    """
    string = "Starting_training@{}\n".format(socket.gethostname())
    actual_time = datetime.datetime.now()
    string += "\t\tStart_hour: {}/{}/{} {}:{}:{}\n".format(actual_time.day, actual_time.month, actual_time.year,
                                                           actual_time.hour, actual_time.minute, actual_time.second)
    if model_name is not None:
        string += "\t\tModel_name: {}\n".format(model_name)
    if kwargs is not None:
        for key, value in kwargs.items():
            string += "\t\t{}: {}\n".format(key, str(value))

    send_message(string)


def notify_training_end(model_name=None, **kwargs):
    """
        Function for notify that the training is ended
        :param model_name: Is the name of the model that has been trained (optional)
        :param kwargs: If you want to send some util information you can put them here
                       as key=value
        :return: None, it will just send the notify
        """
    string = "Training_end@{}\n".format(socket.gethostname())
    actual_time = datetime.datetime.now()
    string += "\t\tEnd_hour: {}/{}/{} {}:{}:{}\n".format(actual_time.day, actual_time.month, actual_time.year,
                                                         actual_time.hour, actual_time.minute, actual_time.second)
    if model_name is not None:
        string += "\t\tModel_name: {}\n".format(model_name)
    if kwargs is not None:
        for key, value in kwargs.items():
            string += "\t\t{}: {}\n".format(key, str(value))

    send_message(string)


def send_image_from_file(image_path, caption=None):
    send_image(open(image_path, 'rb'), caption=caption)


def send_image_from_array(image: np.ndarray, caption=None):
    """
    Sends to telegram an image or a batch of images
    :param image: is the image batch (or single image)
    :param caption: is the message attached to the image
    :return: None
    """
    if len(np.shape(image)) == 4:  # If image is a batch of images
        for im in image:
            imagefile = io.BytesIO()
            Image.fromarray(np.array(im, dtype=np.uint8)).save(imagefile, format='PNG')
            imagefile.read = imagefile.getvalue
            send_image(imagefile, caption=caption)

    elif len(np.shape(image)) == 3:  # If image is a single image
        imagefile = io.BytesIO()
        Image.fromarray(np.array(image, dtype=np.uint8)).save(imagefile, format='PNG')
        imagefile.read = imagefile.getvalue
        send_image(imagefile, caption=caption)


def send_image(image, caption=None):
    bot = telepot.Bot(BOT_TOKEN)
    bot.sendPhoto(CHAT_ID, image, caption=caption)


if __name__ == "__main__":
    h = 200
    w = 200
    img = np.empty(shape=(h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img[i, j, 0] = (i+j)*255//(h+w)
            img[i, j, 1] = ((h-i+w-j)//(h+w))**2 * 255
            img[i, j, 2] = 255*np.sin(i/h * np.pi * 5)
    send_image_from_array(img,
                          caption="Image sent without creating any file! In a slightly cleaner way")
