# --- 通用 MLP 模型 (方案 1, 2) ---
import torch
import torch.nn as nn
import torch.optim as optim
# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# --- 通用训练循环 (方案 1, 2) ---
def train_mlp(model, X_train, y_train, epochs=150, lr=0.001):
    model.to(device)
    # 将 y_train 转换为 (N, 1) 形状
    y_train = y_train.to(device).view(-1, 1)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training on {len(X_train)} samples...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    model.eval()
    return model