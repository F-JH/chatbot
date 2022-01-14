from scripts import HandleData
from config import config
from utils.getTokenizer import getTokenizer

if __name__ == '__main__':
    model_name = "models/chat_DialoGPT_small_zh"
    tokenizer = getTokenizer(model_name)
    dataFile = config.preDataFile
    savePath = config.dataPath

    HandleData.run(tokenizer, dataFile, savePath)