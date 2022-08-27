import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
mean, std = (0.5), (0.5)
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_dataset = torchvision.datasets.MNIST(
    root='.data/',
    train=True,
    download=True,
    transform=transform_train)
train_loader = DataLoader(train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=0,
                          drop_last=True)
test_dataset = torchvision.datasets.MNIST(
    root='.data/',
    train=False,
    download=True,
    transform=transform_test)
test_loader = DataLoader(test_dataset,
                         drop_last=True,
                         batch_size=100,
                         shuffle=False,
                         num_workers=0)

classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
