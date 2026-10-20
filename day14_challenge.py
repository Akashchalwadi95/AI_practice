import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

X_train = torch.randn(size=(300,16))
y_train = torch.randint(low=0, high=4, size=(300,))

class QueryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16,64),
            nn.ReLU(),
            nn.Linear(64,4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = QueryClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 40

print("--- Starting Training Loop ---")
for epoch in range(1, epochs+1):

    model.train()
    optimizer.zero_grad()
    
    logits = model(X_train)

    loss = criterion(logits, y_train)

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:
        predicted = torch.argmax(logits, dim=-1)
        accuracy = (predicted == y_train).float().mean()
        print(f"Epoch [{epoch:2d}/{epochs}] | Loss: {loss.item():.4f} | Accuracy {accuracy.item()*100:.2f}%")
