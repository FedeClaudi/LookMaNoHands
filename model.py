import torch
import torch.nn as nn

class pytorchLSTM(nn.Module):
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