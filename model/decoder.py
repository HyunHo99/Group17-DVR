import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, hidden_dim=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        out = self.fc(x)
        out += identity
        out = self.relu(out)
        
        return out
        

class Decoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=4, hidden_dim=512, depth=5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        
        fc = []
        fc.append(nn.Linear(in_dim, hidden_dim))
        fc.append(nn.ReLU())
        for _ in range(depth):
            fc.append(ResnetBlock())
            
        fc.append(nn.Linear(hidden_dim, out_dim))
        
        self.fc = nn.Sequential(*fc)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, texture=False):
        assert(len(x.shape)==3 and x.shape[-1]==self.in_dim)
        
        out = self.fc(x)
        if(texture):
            return self.sigmoid(out[:,:,1:4])
        else:
            return out[:,:,0]