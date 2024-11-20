# loss for alexnet
from configs import *
def loss(output, target):
    return F.nll_loss(F.log_softmax(output, dim=1), target)