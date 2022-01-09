from config import settings
from utils import MyDataset
from models import transformer
from transformers import AutoTokenizer
from scripts import train
from torch.utils.data import DataLoader

if __name__ == '__main__':
    model_name = "models/bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 增加特殊token
    tokens = {
        "bos_token": "<|BOS|>",
        "eos_token": "<|EOS|>",
        "pad_token": "<|PAD|>",
    }
    tokenizer.add_special_tokens(tokens)
    # 增加query和answer角色的token
    tokenizer.add_tokens(["<|QUE|>", "<|ANS|>"])

    dataset = MyDataset.getDataset(settings.dataPath, tokenizer)
    model = transformer.Transformer(len(tokenizer.get_vocab().keys()), 50, 5, 12)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True)
    train.train(model, 10, dataloader)