import torch


class MSELossNoNaN(torch.nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super(MSELossNoNaN, self).__init__(*args, **kwargs)

    def forward(self, input, target):
        is_not_nan = torch.logical_not(torch.isnan(target))
        input = input[is_not_nan]
        target = target[is_not_nan]
        input = input.type(target.dtype)
        return super(MSELossNoNaN, self).forward(input, target)


class L1LossNoNaN(torch.nn.L1Loss):
    def __init__(self, *args, **kwargs):
        super(L1LossNoNaN, self).__init__(*args, **kwargs)

    def forward(self, input, target):
        is_not_nan = torch.logical_not(torch.isnan(target))
        input = input[is_not_nan]
        target = target[is_not_nan]
        input = input.type(target.dtype)
        return super(L1LossNoNaN, self).forward(input, target)


class L2LossNoNaN(L1LossNoNaN):
    def __init__(self, *args, **kwargs):
        super(L2LossNoNaN, self).__init__(*args, **kwargs)

    def forward(self, input, target):
        return super(L2LossNoNaN, self).forward(input, target)**2