import telepot

BOT_TOKEN = "561223507:AAGvadvBfQcRb3hhTXQN1FN7c2xtn6B9vm0"
CHAT_ID = -307476339


def send_message(message=""):
    bot = telepot.Bot(BOT_TOKEN)
    bot.sendMessage(CHAT_ID, message)


if __name__ == "__main__":
    send_message("TEST")
