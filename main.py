from numpy import inf
from config import settings
from utils import MyDataset
from models import transformer
from torch.utils.data import DataLoader
from config import config
from torch import nn,optim
from utils.getTokenizer import getTokenizer
from scripts.SaveLoad import loadCheckpoint, saveCheckpoint
from utils.schedule import get_cosine_schedule_with_warmup

from scripts.train import valid, predict, train
from tqdm.auto import tqdm
import wandb

def main():
    model_name = "models/chat_DialoGPT_small_zh"
    tokenizer = getTokenizer(model_name)

    trainDataset, validDataset, testDataset = MyDataset.getDataset(settings.dataPath, tokenizer, config.trainConfig)
    model = transformer.Transformer(len(tokenizer.get_vocab().keys()), **config.trainConfig["transformerConfig"])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = getattr(optim, config.trainConfig["optim"])(model.parameters(), lr=config.trainConfig["lr"], **config.trainConfig["params"])

    trainDataLoader = DataLoader(trainDataset, batch_size=config.trainConfig["batch_size"], shuffle=True, pin_memory=True, num_workers=8)
    validDataLoader = DataLoader(validDataset, batch_size=config.trainConfig["batch_size"], shuffle=True, pin_memory=True, num_workers=8)
    print("size of valid:", len(validDataset))

    total_steps = int(-(len(trainDataset) // -config.trainConfig["batch_size"]) * config.trainConfig["n_epochs"])
    warmup_steps = total_steps // 100
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model, optimizer, bestLoss, epoch = loadCheckpoint(model, optimizer, scheduler, config.trainConfig["savepoint"], "cuda")
    train(model, epoch, inf, trainDataLoader, validDataLoader, testDataset, optimizer, criterion, scheduler, tokenizer, config.trainConfig)

    # que, ans = predict(model, testDataset, tokenizer, 3)
    # print("test: [que]{} | [ans]{}".format("".join(que), "".join(ans)))

print(__name__)
if __name__ == '__main__':
    main()