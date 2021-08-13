import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision
from collections import OrderedDict
from torch import nn
from torchvision import datasets, transforms, models


def add_argparse_arguments():
    parser.add_argument('input', help='??', type=str)
    parser.add_argument('checkpoint', help='Provide directory to the checkpoint. Mandatory argument', type=str)
    parser.add_argument('--topk', help='Provide how many top classes you want the model to predict. Optional argument',
                        type=int)
    parser.add_argument('--gpu', help="Option to use GPU", type=str)
    parser.add_argument('--category_names',
                        help="Option to use a file with mapping of categories to real names. Optional argument",
                        type=str)


def load_data(data_dir, which_data):
    """Given the general data directory and the type of data needed (i.e. 'train', 'valid' or 'test') it returns a data loader"""
    sub_dir = data_dir + f'/{which_data}'

    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    # TODO: Load the datasets with ImageFolder
    imagefolder_dataset = torchvision.datasets.ImageFolder(train_dir, transform=data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloader = torch.utils.data.DataLoader(imagefolder_dataset, batch_size=32, shuffle=True)

    return dataloader


def load_model(file_path):
    checkpoint = torch.load(file_path, map_location=map_location)
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']

    for param in model.parameters():
        param.requires_grad = False

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = PIL.Image.open(image)

    start_width, start_height = img.size

    if start_width < start_height:
        size = [256, 256 ** 600]
    else:
        size = [256 ** 600, 256]

    img.thumbnail(size)

    center = start_width / 4, start_height / 4
    left, top, right, bottom = center[0] - (244 / 2), center[1] - (244 / 2), center[0] + (244 / 2), center[1] + (
                244 / 2)
    img = img.crop((left, top, right, bottom))

    numpy_img = np.array(img) / 255

    # Normalize each color channel
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    numpy_img = (numpy_img - mean) / std

    # Set the color to the first channel
    numpy_img = numpy_img.transpose(2, 0, 1)

    return numpy_img


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    processed_image = process_image(image_path)

    im = torch.from_numpy(processed_image).type(torch.FloatTensor)
    im = im.unsqueeze(dim=0)

    with torch.no_grad():
        output = model.forward(im)
    output_probabilities = torch.exp(output)

    probabilities, indexes = output_probabilities.topk(topk)
    probabilities = probabilities.numpy().tolist()[0]
    indexes = indexes.numpy().tolist()[0]

    mapping = {val: key for key, val in
               model.class_to_idx.items()
               }

    classes = [mapping[item] for item in indexes]
    classes = np.array(classes)

    return probabilities, classes


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.array(image)
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


If
__name__ == "__main__":
# initiate argument parser
parser = argparse.ArgumentParser(description='Training job argument parser')
add_argparse_arguments()
args = parser.parse_args()

if args.GPU:
    device = set_device_to_cuda()

if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'

model = load_model('checkpoint.pth')

# load data for testing???
testloader = load_data(args.data_dir, 'test')

process_image
