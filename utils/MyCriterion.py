from torch import nn

class MyCriterion(nn.Module):
    def __init__(self, ignore_token):
        super(MyCriterion, self).__init__()
        self.criterion == nn.CrossEntropyLoss()
        self.ignore_token = ignore_token
