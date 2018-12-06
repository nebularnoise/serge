from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import MNIST, FashionMNIST

def load_MNIST(dataset_name, batch=128):
    DATASET = MNIST if dataset_name == "MNIST" else FashionMNIST

    trainset = DATASET(root='./%s/'%dataset_name,
                    train=True,
                    transform=transforms.ToTensor(),
                    download=True)
    testset = DATASET(root='./%s/'%dataset_name,
                    train=False,
                    transform=transforms.ToTensor(),
                    download=True)
    labels = ['#0','#1','#2','#3','#4','#5','#6','#7','#8','#9']

    N_train = len(trainset)
    N_test = len(testset)

    train_loader = DataLoader(dataset=trainset,
                          batch_size=batch,
                          shuffle=True)

    test_loader = DataLoader(dataset=testset,
                         batch_size=batch,
                         shuffle=False)
    return train_loader, test_loader
