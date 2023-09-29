import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torchvision.datasets import ImageFolder
from alectio_sdk.sdk.alectio_dataset import AlectioDataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# prepare dataset
# paste your experiment token here
alectio_dataset = AlectioDataset(token="<YOUR EXPERIMENT TOKEN", root="./data", framework="pytorch")

# train dataset
train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
train_dataset, train_dataset_len, train_class_to_idx = alectio_dataset.get_dataset(
    dataset_type="train", transforms=train_transforms
)
# number of class in the dataset
num_classes = len(train_class_to_idx.keys())
# test dataset
test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)





test_dataset, test_dataset_len, test_class_to_idx = alectio_dataset.get_dataset(
    dataset_type="test", transforms=test_transforms
)
TRAIN_DATASET = ImageFolder('data/train', transform=train_transforms)
TEST_DATASET = ImageFolder('data/test',transform=test_transforms)
num_images = len(TRAIN_DATASET)
labeled_images = list(range(num_images))

# prepare pre-trained model
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, num_classes)


# parameters
epoch = 10
batch_size = 32
lr = 0.0001
# if you don't have a GPU machine uncomment the line 55 and comment line 56
device = torch.device("cpu")
# device = torch.device("cuda")
# loss function and criterion
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(
    model_conv.parameters(),
    lr=lr,
)


# SDK Processes

def getdatasetstate(args):
    return {k: k for k in range(train_dataset_len)}


def train(args, labeled, resume_from, ckpt_file):
    print("inside the traininggggg.....")
    # if you get index of out of bound error please uncomment the line below and run your code
    labeled = [x-1 for x in labeled]

    labeled_dataset = Subset(train_dataset, labeled)
    # if you don't have a GPU machine remove num_workers argument
    train_dataloader = DataLoader(
        dataset=labeled_dataset, batch_size=batch_size, shuffle=True
    )
    model_conv.train()
    print(epoch)
    for i in range(epoch):
        print(f'Training started for epoch {i}...')
        for data in tqdm(train_dataloader, desc="Training"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model_conv(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(args, ckpt_file):
    # if you don't have a GPU machine remove num_workers argument
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    predictions, targets = [], []
    model_conv.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Testing"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model_conv(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return {"predictions": predictions, "labels": targets}


def infer(args, unlabeled, ckpt_file):
    unlabeled = Subset(train_dataset, unlabeled)
    # if you don't have a GPU machine remove num_workers argument
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled, batch_size=4, shuffle=False
    )

    model_conv.eval()

    correct, total, k = 0, 0, 0
    outputs_fin = {}
    for i, data in tqdm(enumerate(unlabeled_loader), desc="Inferring"):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model_conv(images).data

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for j in range(len(outputs)):
            outputs_fin[k] = {}
            outputs_fin[k]["prediction"] = predicted[j].item()
            outputs_fin[k]["pre_softmax"] = outputs[j].cpu().numpy().tolist()
            k += 1

    return {"outputs": outputs_fin}

if __name__ == '__main__':
    train(args=None, labeled=labeled_images, resume_from=None, ckpt_file=None)
    test(args=None,ckpt_file='ckpt_0')
    # infer(args=None, unlabeled=[10,20,30],ckpt_file='ckpt_0')
