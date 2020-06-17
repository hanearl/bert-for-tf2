import telegram

TELEGRAM_TOKEN = '1218931377:AAFFfNAjyf058wKpTAVjjhsz5inhvCRuwFQ'


class ExamAlarmBot:
    def __init__(self):
        self.bot = telegram.Bot(token=TELEGRAM_TOKEN)

    def send_msg(self, message):
        self.bot.sendMessage(chat_id=813359225, text=message)
