import telegram

TELEGRAM_TOKEN = '1218931377:AAFFfNAjyf058wKpTAVjjhsz5inhvCRuwFQ'


class ExamAlarmBot:
    def __init__(self):
        self.bot = telegram.Bot(token=TELEGRAM_TOKEN)

    def send_msg(self, message):
        updates = self.bot.getUpdates()
        chat_id = updates[-1].message.chat_id

        self.bot.sendMessage(chat_id=chat_id, text=message)
