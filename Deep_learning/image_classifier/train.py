import argparse
import json
import torch
import torchvision
from torch import nn
from torchvision import transforms, models

def load_data(data_dir, which_data):
    """Given the general data directory and the type of data needed (i.e. 'train', 'valid' or 'test') returns a data loader"""
    sub_dir = data_dir + f'/{which_data}'

    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    #Load the datasets with ImageFolder
    imagefolder_dataset = torchvision.datasets.ImageFolder(sub_dir, transform=data_transforms)

    #Defines the dataloaders
    dataloader = torch.utils.data.DataLoader(imagefolder_dataset, batch_size=32, shuffle=True)

    return dataloader

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def add_argparse_arguments():
    """Adds all optional and mandatory arguments to the parser with a helper text and default values"""
    parser.add_argument('data_dir',
                        action="store",
                        help='Provide data directory. Mandatory argument',
                        type=str)
    parser.add_argument('--learning_rate',
                        action="store",
                        help='Provide learning rate at which to train the model. Optional argument',
                        default=0.001,
                        type=float)
    parser.add_argument('--hidden_units',
                        action="store",
                        help='Provide the number of hidden units to train your network with. Optional argument',
                        default=120,
                        type=int)
    parser.add_argument('--epochs',
                        action="store",
                        help='Provide the number of epochs for training process. Optional argument',
                        default=12,
                        type=int)
    parser.add_argument('--save_dir',
                        action="store",
                        help='Provide saving directory for your checkpoint. Optional argument',
                        default= "./checkpoint.pth",
                        type=str)
    parser.add_argument('--arch',
                        action="store",
                        help='Alexnet can be used if this argument specified, otherwise VGG16 will be used',
                        default="vgg16",
                        type=str)
    parser.add_argument('--gpu',
                        action="store",
                        help="Option to use GPU",
                        default="cpu",
                        type=str)


def set_device_to_cuda():
    """Sets device to GPU if specified by --gpu flag"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def initiate_model(pre_trained_arch="vgg16", hidden_units):

    #     TODO find a way to diversify
    if pre_trained_arch = "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(25088, 3264),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(3264, 120),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),
                                     nn.Linear(120, 80),
                                     nn.ReLU(),
                                     nn.Linear(80, 40),
                                     nn.ReLU(),
                                     nn.Linear(40, 102),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr)

    model.to(device)


def train_model(model, trainloader, validationloader, epochs=12, lr=0.001, print_every=40):
    """Prints out training loss, validation loss, and validation accuracy as the network trains"""

    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # move inputs and labels to defult device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model.forward(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"valid loss: {valid_loss / len(validationloader):.3f}.. "
                      f"valid accuracy: {accuracy / len(validationloader):.3f}")

                running_loss = 0
                model.train()


def save_checkpoint(model, filepath='checkpoint.pth'):
    """"""
    model.class_to_idx = train.class_to_idx

    checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'mapping': model.class_to_idx}

    torch.save(checkpoint, filepath)


if __name__ == "__main__":
# initiate argument parser
parser = argparse.ArgumentParser(description='Training job argument parser')
add_argparse_arguments()
args = parser.parse_args()

# load data for training
trainloader = load_data(args.data_dir, 'train')
validationloader = load_data(args.data_dir, 'valid')

if args.GPU:
    device = set_device_to_cuda()

# TODO: if --arch is given, else use vgg 16 as default
if args.arch:
    model = initiate_model(args.arch)

print('Training Starting')

#     check if I need conditions here
train_model(model, trainloader, validationloader, args.epochs, args.learning_rate)

save_checkpoint(model, args.save_dir)

print("Training Done")
