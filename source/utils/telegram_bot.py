import telepot
import datetime
import socket
import numpy as np
import shutil
from data_manager import path_manager as pm
from image_loader.image_loader import save_image_from_matrix

BOT_TOKEN = "561223507:AAGvadvBfQcRb3hhTXQN1FN7c2xtn6B9vm0"
CHAT_ID = -307476339


def send_message(message=""):
    bot = telepot.Bot(BOT_TOKEN)
    bot.sendMessage(CHAT_ID, message)


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


def send_image_from_file(image_path):
    send_image(open(image_path, 'rb'))


def send_image_from_array(image: np.ndarray):
    path = pm.resources_path("tmp", "tmp.png")
    save_image_from_matrix(image, path)
    send_image_from_file(path)
    shutil.rmtree(pm.resources_path("tmp"))


def send_image(image):
    bot = telepot.Bot(BOT_TOKEN)
    print(type(image))
    bot.sendPhoto(CHAT_ID, image)


if __name__ == "__main__":
    send_image_from_array(np.zeros(shape=[200, 200, 3]))
