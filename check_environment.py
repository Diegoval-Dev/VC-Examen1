import torch
import torchvision
import numpy as np
import platform
import sklearn
import cv2
import torch
import torch.nn as nn

print("==== SYSTEM INFO ====")
print("Platform:", platform.platform())
print("Processor:", platform.processor())

print("\n==== TORCH INFO ====")
print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

print("\nCUDA available:", torch.cuda.is_available())
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

print("\n==== NUMERICAL TEST ====")
x = torch.randn(1000, 1000).to(device)
y = torch.matmul(x, x)
print("Matrix multiplication successful on:", y.device)

print("\n==== LIBRARIES ====")
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("OpenCV:", cv2.__version__)

print("\nEnvironment check completed successfully.")


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = nn.Linear(10, 2).to(device)
x = torch.randn(32, 10).to(device)
y = torch.randint(0, 2, (32,)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

outputs = model(x)
loss = criterion(outputs, y)

loss.backward()
optimizer.step()

print("Forward + Backward successful on:", device)