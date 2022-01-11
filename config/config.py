trainConfig = {
    "n_epochs": 1,
    "log": False,
    "lr": 5e-6,
    "batch_size": 16,
    "optim": "Adam",
    "params": {
        "betas": (0.9, 0.99),
        "eps": 1e-8,
    },
    "transformerConfig":{
        "d_model": 60,
        "n_head": 3,
        "num_of_layer": 8
    },
    "savepoint": "checkpoint/checkpoint.pth"
}