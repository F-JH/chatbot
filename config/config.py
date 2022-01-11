trainConfig = {
    "n_epochs": 1,
    "log": True,
    "lr": 5e-6,
    "batch_size": 128,
    "rest_batch": 500,
    "optim": "Adam",
    "dataset": {
        # 这个计算参考 utils/MyDataset.py 87行，不想使用百分比来控制valid和test集大小，各位不喜欢的话可以自行修改代码
        "validSplit": 430,
        "testSplit": 500
    },
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