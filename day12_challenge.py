import torch

w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

x = torch.tensor([2.0])
y_true = torch.tensor([10.0])

learning_rate = 0.05

for epoch in range(20):
    y_pred = torch.dot(w,x) + b

    loss = (y_true-y_pred)**2
   
    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        w.grad.zero_()
        b.grad.zero_()

    if epoch % 5 == 0:
        print(f"Epoch: {epoch:2d}")
        print(f"Prediction: {y_pred.item():.4f}")
        print(f"Loss: {loss.item():.4f}")

print(f"\nFinal weight: {w.item():.4f}")
print(f"Final bias:   {b.item():.4f}")