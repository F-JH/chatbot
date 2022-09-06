import torch
from utils import TransformerRuntimeMask
from utils.getTokenizer import getTokenizer
from config import config

if __name__ == '__main__':
    model_name = "models/chat_DialoGPT_small_zh"
    tokenizer = getTokenizer(model_name)
    model = TransformerRuntimeMask.Transformer(len(tokenizer.get_vocab().keys()), tokenizer.pad_token_id, **config.trainConfig["transformerConfig"])
    torchScript = torch.jit.script(model)
    print(torchScript.code)