import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
from pathlib import Path

import model


# Define your dataset
class CustomDataset(Dataset):
    def __init__(self, ):
        data_path = Path("data/processed")
        csv_files = list(data_path.glob("*.csv"))

        self.Ys = []
        self.Xs = []

        for f in csv_files:
            df = pd.read_csv(f)

            # interpolate missing values
            df = df.interpolate(method="linear", limit_direction="both")

            # make sure there are no nans
            if any(df.isna().any()):
                print(f"Skipping {f} due to NaNs")
                print(df.head(5))
                continue

            # trim to 1800 samples
            df = df.iloc[:1800]
            
            # chunk into 600 sample segments
            for i in (1, 2, 3):
                _df = df.iloc[(i-1)*600:i*600]

                # extract features
                self.Ys.append(_df[["cursor_x", "cursor_y"]].values.astype(np.float32))
                self.Xs.append(_df.drop(columns=["cursor_x", "cursor_y"]).values.astype(np.float32))


        self.data = list(zip(self.Xs, self.Ys))

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


hidden_size = 128
lr = 1e-2
num_epochs = 500

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the dataset and data loader
dataset = CustomDataset()

n_in = dataset.data[0][0].shape[1]
n_out = dataset.data[0][1].shape[1]

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

rnn = model.pytorchLSTM(n_in, hidden_size, n_out).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(rnn.parameters(), lr=lr)

hidden = None
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, (X, Y) in enumerate(dataloader):
        optimizer.zero_grad()

        # Detach the hidden state from the previous sequence
        if hidden is not None:
            hidden = (hidden[0].detach(), hidden[1].detach())

        # Forward pass
        outputs, hidden = rnn(X, hidden=hidden)
        loss = criterion(outputs, Y)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        epoch_loss += loss.item()
        
    if epoch % 10 == 0 or epoch == 0:
        print(f'Epoch [{epoch}/{num_epochs}],  Loss: {epoch_loss:.4f}')

# Save the trained rnn
torch.save(rnn.state_dict(), 'models/model.pt')