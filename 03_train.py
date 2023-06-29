import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


import model


hidden_size = 256
lr = 1e-4
num_epochs = 50_000

# Set device
if not torch.cuda.is_available():
    raise Exception('GPU not found.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Create the dataset and data loader
dataset = model.CustomDataset()

n_in = dataset.data[0][0].shape[1]
n_out = dataset.data[0][1].shape[1]

dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# rnn = model.LSTM(n_in, hidden_size, n_out).to(device)
mlp = model.MLP(n_in, n_out).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp.parameters(), lr=lr)

hidden = None
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (X, Y) in enumerate(dataloader):
        optimizer.zero_grad()

        # Detach the hidden state from the previous sequence
        if hidden is not None:
            hidden = (hidden[0].detach(), hidden[1].detach())

        # Forward pass
        # outputs, hidden = rnn(X, hidden=hidden)
        outputs = mlp(X)
        loss = criterion(outputs, Y)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        epoch_loss += loss.item()
        
    if epoch % 100 == 0 or epoch == 0:
        print(f'Epoch [{epoch}/{num_epochs}],  Loss: {epoch_loss:.4f}')

# Save the trained rnn
torch.save(mlp.state_dict(), 'models/mlp.pt')
# torch.save(rnn.state_dict(), 'models/rnn.pt')