import torch.nn as nn

# defining the model
class ODESolver(nn.Module):
    def __init__(self):
        super(ODESolver, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.net(x)