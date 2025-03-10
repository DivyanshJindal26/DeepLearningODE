import torch
import torch.optim as optim
from model import ODESolver

# defining the loss function to refine the model
def loss_function(model, x):
    x.requires_grad = True
    y = model(x)
    
    dy_dx = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    
    loss = torch.mean((dy_dx - x) ** 2)
    return loss

# the model and optimizer
model = ODESolver()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# values of x to train.
x_train = torch.linspace(-2, 2, 100).view(-1, 1)

# training the model for 20000 epochs
for epoch in range(20000):
    optimizer.zero_grad()
    loss = loss_function(model, x_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print("Training complete!")

# saving the model
torch.save(model.state_dict(), "ode_x.pth")
print("Model saved successfully!")
