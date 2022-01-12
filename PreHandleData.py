from scripts import HandleData
from utils.getTokenizer import getTokenizer

if __name__ == '__main__':
    model_name = "models/chat_DialoGPT_small_zh"
    tokenizer = getTokenizer(model_name)
    HandleData.run(tokenizer)