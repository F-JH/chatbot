import torch

def saveCheckpoint(model, optimizer, scheduler, bestLoss, epoch, path):
    model.eval()
    save_dict = {
        "epoch": epoch,
        "bestLoss": bestLoss,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict()
    }
    torch.save(save_dict, path)

def loadCheckpoint(model, optimizer, scheduler, path, device):
    modelCKPT = torch.load(path, device)
    model.load_state_dict(modelCKPT["state_dict"])
    optimizer.load_state_dict(modelCKPT["optimizer"])
    epoch = modelCKPT["epoch"]
    bestLoss = modelCKPT["bestLoss"]
    scheduler.load_state_dict(modelCKPT["scheduler"])
    model.eval()
    return model, optimizer, bestLoss, epoch