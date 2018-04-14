import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..")))

import datetime
import socket
import utils.telegram_bot as tele

if __name__ == "__main__":
    string = "###### System online ######\n"
    string += "\t\t" + socket.gethostname() + "\n"
    actual_time = datetime.datetime.now()
    string += "\t\tat time: {}/{}/{} {}:{}:{}\n".format(actual_time.day, actual_time.month, actual_time.year,
                                                        actual_time.hour, actual_time.minute, actual_time.second)
    string += "\t\tHey! I'm here, remember to shutdown me!\n"
    tele.send_message(string, disable_notification=True)
