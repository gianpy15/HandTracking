import telepot
import datetime
import socket
import numpy as np
import io
import time
import requests
from PIL import Image

URL = 'https://api.telegram.org/bot{0}/{1}'
BOT_TOKEN = "561223507:AAGvadvBfQcRb3hhTXQN1FN7c2xtn6B9vm0"
CHAT_ID = -307476339


def newbot():
    return telepot.Bot(BOT_TOKEN)


def send_message(message="", bot=None, disable_notification=False):
    if bot is None:
        __send_message(message=message)
    else:
        bot.sendMessage(CHAT_ID, message, disable_notification=disable_notification)


def __send_message(message=""):
    response = requests.post(
        url=URL.format(BOT_TOKEN, 'sendMessage'),
        data={'chat_id': CHAT_ID, 'text': message}
    ).json()

    return response


def notify_training_starting(bot=None, model_name=None, **kwargs):
    """
    Function for notify that the training is starting
    :param bot: The bot used for sending the message; if None, a new one is created
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

    send_message(bot=bot, message=string)


def notify_training_end(bot=None, model_name=None, **kwargs):
    """
        Function for notify that the training is ended
        :param bot: The bot used for sending the message; if None, a new one is created
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

    send_message(bot=bot, message=string)


def send_image_from_file(image_path, bot=None, caption=None):
    send_image(open(image_path, 'rb'), bot=bot, caption=caption)


def send_image_from_array(image: np.ndarray, bot=None, caption=None):
    """
    Sends to telegram an image or a batch of images
    :param bot: The bot used for sending the message; if None, a new one is created
    :param image: is the image batch (or single image)
    :param caption: is the message attached to the image
    :return: None
    """
    if len(np.shape(image)) == 4:  # If image is a batch of images
        for im in image:
            imagefile = io.BytesIO()
            Image.fromarray(np.array(im, dtype=np.uint8)).save(imagefile, format='PNG')
            imagefile.read = imagefile.getvalue
            send_image(image=imagefile, bot=bot, caption=caption)

    elif len(np.shape(image)) == 3:  # If image is a single image
        imagefile = io.BytesIO()
        Image.fromarray(np.array(image, dtype=np.uint8)).save(imagefile, format='PNG')
        imagefile.read = imagefile.getvalue
        send_image(image=imagefile, bot=bot, caption=caption)


def send_image(image, bot=None, caption=None):
    if bot is None:
        bot = newbot()
    bot.sendPhoto(CHAT_ID, image, caption=caption)


if __name__ == "__main__":
    bot = newbot()
    send_message(message="test senza usare telepot")

    #time.sleep(301)

    #send_message(bot, "test after 300 secs")
    """
    h = 200
    w = 200
    img = np.empty(shape=(h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img[i, j, 0] = (i+j)*255//(h+w)
            img[i, j, 1] = ((h-i+w-j)//(h+w))**2 * 255
            img[i, j, 2] = 255*np.sin(i/h * np.pi * 5)
    send_image_from_array(img,
                          bot,
                          caption="Image sent without creating any file! In a slightly cleaner way")
    """
