import torch
import matplotlib.pyplot as plt
from model import ODESolver

# loading the trained models
model_loaded = ODESolver()
model_loaded.load_state_dict(torch.load("ode_x.pth")) # change the file path as needed
model_loaded.eval()

# the values for graphing
x_test = torch.linspace(-2, 2, 100).view(-1, 1)
y_pred = model_loaded(x_test).detach().numpy()

# initial condition
x_0 = 0
y_0 = 0  

# calculating the constant c
y_x0 = model_loaded(torch.tensor([[x_0]], dtype=torch.float32)).item()
C = y_0 - y_x0

y_pred += C

# plotting the graph
plt.plot(x_test.numpy(), y_pred, label="Neural Network Prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
