import torch
import torch.nn as nn
import torch.nn.functional as F
from Dataloader_mnist import train_loader, test_loader


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel

        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # # an affine operation: y = Wx + b
        self.quant = torch.quantization.QuantStub()
        self.fc1 = nn.Linear(28 * 28, 2048)  # 5*5 from image dimension
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 4096)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4096, 10)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.quant(x)
        x = torch.flatten(x, 1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)

        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # # If the size is a square, you can specify with a single number
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x



epoch = 1
lr = 0.01
cpu_device = torch.device("cpu:0")
device = torch.device("cpu:0")
net_fp32 = Net().to(device)

def train(model, data_loader, optimizer, epoch):
    running_loss = 0.0
    model.train()
    criterion = nn.CrossEntropyLoss()
    for i, data in enumerate(data_loader):
        model.zero_grad()
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs.to(device), labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(data_loader.dataset)
    print("Epoch {:<5} Train loss: {:.4f}".format(epoch, epoch_loss))


def test(model, data_loader, epoch):
    count = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_corrects = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            count += inputs.size(0)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
        epoch_acc = running_corrects / count
        epoch_loss = running_loss / count
        print("Epoch {:<5} ACC: {:.2f}% Test Loss: {:.5f}".format(epoch, epoch_acc * 100, epoch_loss))
    return "{:.2f}%".format(epoch_acc * 100)


net_fp32.train()
net_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
net_fp32_fused = torch.quantization.fuse_modules(net_fp32, [['fc1', 'relu1'], ['fc2', 'relu2']])



net_fp32_prepared = torch.quantization.prepare_qat(net_fp32_fused)

# load state dict from file
print('==>')
print('load fp32 model state dict from file')
net_fp32_prepared.load_state_dict(torch.load('./saved_model/qtrainfff_fp32.ckpt'))

net_fp32_prepared.train()
# training loop
optimizer = torch.optim.SGD(net_fp32_prepared.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
for i in range(epoch):
    print("=" * 100)
    train(net_fp32_prepared, train_loader, optimizer, epoch=i)
    test(net_fp32_prepared, test_loader, epoch=i)


# to cpu device
# device = cpu_device
# net_fp32_prepared.to(cpu_device)
net_fp32_prepared.eval()
net_int8 = torch.quantization.convert(net_fp32_prepared)
print(f"convert to int8 type successfully")



acc = test(net_int8, test_loader, epoch)
print(f"Quantized Model Accuracy: {acc}")

# state_dict = net_int8.state_dict()
# torch.save(net_fp32_prepared.state_dict(), './saved_model/qtrainfff_fp32.ckpt')
# torch.jit.save(torch.jit.script(net_int8), './saved_model/qtrainfff_int8.ckpt')
