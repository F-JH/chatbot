import torch
from tqdm.auto import tqdm
import json, wandb
from scripts.predict import predict
from scripts.SaveLoad import saveCheckpoint

def valid(model, criterion, validData):
    model.eval()
    validLoss = 0
    n = 0
    for batch in tqdm(validData):
        n += 1
        batch = [i.to("cuda") for i in batch]
        with torch.no_grad():
            output = model(batch[0], batch[1])
            loss = criterion(output, batch[2].view(-1))
            validLoss += loss.item()
    validLoss /= n
    return validLoss

def trainRuntimeMask(model, epoch, batch_n, bestLoss, trainData, validData, testData, optimizer, criterion, scheduler, tokenizer, config):
    use_load_batch_n = True
    if config["log"]:
        wandb.init()
    count = 0
    while epoch < config["n_epochs"]:
        model.train()
        totalLoss = 0
        for data in tqdm(trainData):
            if use_load_batch_n and count < batch_n:
                count += 1
                continue
            else:
                use_load_batch_n = False
            batch_n += 1
            optimizer.zero_grad()
            data = [i.to("cuda") for i in data]
            output = model(data[0], data[1])
            loss = criterion(output, data[2].view(-1))
            totalLoss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 等不了一个epoch了，太多了
            if batch_n % config["rest_batch"] == 0:
                # print("start valid...")
                totalLoss /= config["rest_batch"]
                validLoss = valid(model, criterion, validData)
                if config["log"]:
                    log = {
                        "train loss": totalLoss,
                        "valid loss": validLoss
                    }
                    wandb.log(log)
                msg = "epoch : {}| Train Loss: {}| Valid Loss: {}".format(epoch, totalLoss, validLoss)
                print(msg)
                que, ans, label, n = predict(model, testData, tokenizer, config["transformerConfig"]["n_head"])
                que = "".join(que).replace(tokenizer.pad_token, "")
                ans = "".join(ans).replace(tokenizer.pad_token, "")
                label = "".join(label).replace(tokenizer.pad_token, "")
                print("test:{}\n[que]{}\n[ans]{}\n[label]{}".format(n, que, ans, label))
                if validLoss < bestLoss:
                    bestLoss = validLoss
                    print("save model................................................")
                    saveCheckpoint(model, optimizer,scheduler, bestLoss, epoch, batch_n, config["savepoint"])
                totalLoss = 0
        batch_n = 0