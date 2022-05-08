import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split
from surrogate_model import ResNet2DFramewise
from data_loader import get_default_device, DeviceDataLoader, to_device
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm

random_seed = 42
torch.manual_seed(random_seed);

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomHorizontalFlip(0.4),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Resize((64, 64))
     ])

batch_size = 64
checkpoint_path = "checkpoint/"
device = get_default_device()

train_ds = datasets.ImageFolder("train_dataset_frames", transform=transform)
val_ds = datasets.ImageFolder("val_dataset_frames", transform=transform)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
                                           pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)

model = ResNet2DFramewise()
model = to_device(model, device)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in tqdm(range(epochs)):
        print(f"Starting for epoch: {epoch} at {time.strftime('%X %x')}")
        # Training phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        scheduler.step()
        state_dict = {"model": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict()}
        torch.save(state_dict, checkpoint_path + "GreyBox_P1_" + str(epoch) + ".pt")
    return history


num_epochs = 10
opt_func = torch.optim.SGD
lr = 0.001

history = fit(num_epochs, lr, model, train_loader, val_loader, opt_func)
print(history)
