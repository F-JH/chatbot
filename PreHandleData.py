import os
from scripts import HandleData
from config import config
from utils.getTokenizer import getTokenizer

if __name__ == '__main__':
    model_name = "models/chat_DialoGPT_small_zh"
    tokenizer = getTokenizer(model_name)
    dataFile = config.preDataFile
    savePath = config.dataPath

    if not os.path.exists(savePath):
        os.mkdir(savePath)
    HandleData.run(tokenizer, dataFile, savePath)