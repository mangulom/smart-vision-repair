import torch
from torchvision import datasets, transforms
from torch import nn

# -------------------------
# Transformaciones
# -------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# -------------------------
# Dataset
# -------------------------
dataset = datasets.ImageFolder("training", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

num_classes = len(dataset.classes)
print("Clases detectadas:", dataset.classes)
print("Número de clases:", num_classes)

# -------------------------
# Modelo dinámico
# -------------------------
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128*128*3, 256),
    nn.ReLU(),
    nn.Linear(256, num_classes)   # 🔥 dinámico
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# -------------------------
# Entrenamiento
# -------------------------
def train():
    for epoch in range(5):
        total_loss = 0

        for images, labels in loader:
            preds = model(images)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "models/defect_model.pt")
    print("Modelo entrenado y guardado correctamente.")

if __name__ == "__main__":
    train()
