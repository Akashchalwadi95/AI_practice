import torch
import torch.nn as nn

class IntentRouterNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_intents: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden_size, num_intents)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)

        logits = self.out(x)
        return logits

model = IntentRouterNet(input_size=64, hidden_size=128, num_intents=4)

X_batch = torch.randn(size=(8,64))

logits = model(X_batch)

probabilities = torch.softmax(logits, dim=-1)
predicted_classes = torch.argmax(probabilities, dim=-1)

print(f"Predicted: {predicted_classes}")
print(f"Output shape: {logits.shape}")
print("Probabilities shape:", probabilities.shape)

