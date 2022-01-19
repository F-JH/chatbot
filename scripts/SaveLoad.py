import torch

def saveCheckpoint(model, optimizer, scheduler, bestLoss, epoch, batch_n, path):
    model.eval()
    save_dict = {
        "epoch": epoch,
        "batch_n": batch_n,
        "bestLoss": bestLoss,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }
    torch.save(save_dict, path)

def loadCheckpoint(model, optimizer, scheduler, path, device):
    print("load {}".format(path))
    modelCKPT = torch.load(path, device)
    model.load_state_dict(modelCKPT["state_dict"])
    optimizer.load_state_dict(modelCKPT["optimizer"])
    epoch = modelCKPT["epoch"]
    bestLoss = modelCKPT["bestLoss"]
    batch_n = modelCKPT.get("batch_n") if modelCKPT.get("batch_n") else 0
    # scheduler.load_state_dict(modelCKPT["scheduler"])
    model.eval()
    return model, optimizer, bestLoss, epoch, batch_n, scheduler

def loadWeight(model, path, device):
    print("load {}".format(path))
    modelCKPT = torch.load(path, device)
    model.load_state_dict(modelCKPT["state_dict"])
    return model

def saveModule(model):
    model.eval()
    torch.save(model, "model.pt")