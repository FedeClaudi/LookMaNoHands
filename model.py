import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
from pathlib import Path

# Define your dataset
class CustomDataset(Dataset):
    def __init__(self, ):
        data_path = Path("data/processed")
        csv_files = list(data_path.glob("*.csv"))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        self.Xs = [torch.from_numpy(x).to(device) for x in self.Xs]
        self.Ys = [torch.from_numpy(y).to(device) for y in self.Ys]
        self.data = list(zip(self.Xs, self.Ys))

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


class LSTM(nn.Module):
    def __init__(self, n_in, hidden_size, n_out):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_in = n_in
        self.lstm = nn.LSTM(n_in, hidden_size, num_layers=1, batch_first = True)
        self.output_layer = nn.Linear(hidden_size, n_out)

        
    def forward(self, X, hidden = None):
        if hidden == None:
            b =  X.shape[0]
            if len(X.shape) == 3:
                hidden = (
                    torch.zeros(1, b, self.hidden_size).to(X.device),
                    torch.zeros(1, b, self.hidden_size).to(X.device)
                )
            else:
                hidden = (
                    torch.zeros(1, self.hidden_size).to(X.device),
                    torch.zeros(1, self.hidden_size).to(X.device)
                )
            out, hidden = self.lstm(X, hidden)
            out = self.output_layer(out)
        else:
            out, hidden = self.lstm(X, hidden)
            out = self.output_layer(out)
        return out, hidden
    

class MLP(nn.Module):
    def __init__(self, n_in, n_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, n_out)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x