import chatbot.scripts.ingest as ingest
import chatbot.scripts.chat_bot as bot


def start_bot():
    ingest.create_vectors()
    bot.start_chat_bot()


if __name__ == "__main__":
    start_bot()
