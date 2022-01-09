from tqdm.auto import tqdm

def train(model, n_epoch, dataloader):
    for epoch in range(n_epoch):
        for data in tqdm(dataloader):
            data = [i.to("cuda") for i in data]
            output  = model(data[0], data[2])
            break
        break