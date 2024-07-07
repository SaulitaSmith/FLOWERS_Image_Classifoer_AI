import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
import argparse
import os
from pathlib import Path

def argument_parser():
    # creating ArgumentParser object
    parser = argparse.ArgumentParser(description='Train a deep learning model')

    # adding arguments
    parser.add_argument("--data_dir",
                        type = str,
                        default = 'flowers/',
                        help='Path to folder of flowers_images')

    parser.add_argument('--learning_rate',
                        type = float,
                        default = 1e-3,
                        help='learning rate for training')

    parser.add_argument('--hidden_units',
                        type = int,
                        default = 512,
                        help ='number of hidden units in the classifier')

    parser.add_argument('--epochs',
                        type = int,
                        default = 5,
                        help ='number of training epochs')

    parser.add_argument('--save_dir',
                        type = str,
                        default = 'saved_model',
                        help ='Path to the directory for saved models')

    parser.add_argument('--arch',
                        type = str,
                        default = 'resnet34',
                        choices = ('resnet18', 'resnet34', "resnet50"),
                        help ='NN architectures to choose from')

    parser.add_argument('--gpu',
                        action="store_true",
                        help = 'Switch to GPU for optimal performance')

    # parsing arguments
    args = parser.parse_args()

    # returning args
    return args

# function for training the model
def train_model(arch, data_dir, save_dir, learning_rate, hidden_units, epochs, use_gpu):

    # defining device
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

    # using device check
    print(f"Using device: {device}")
    print()

    # Define the transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size=(256,256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=(256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=(256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(f'{data_dir}/{x}', transform=data_transforms[x])
        for x in ['train', 'valid', 'test']
    }

    # Define the dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], 
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=os.cpu_count() if str(device)=="cuda:0" and use_gpu else 0,
                                       pin_memory=True if str(device)=="cuda:0" and use_gpu else False)
        for x in ['train', 'valid', 'test']
    }

    # Load the pre-trained model
    model = getattr(models, arch)(pretrained=True)

    # Freeze the pre-trained model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier with a new one
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, hidden_units),
                             nn.ReLU(),
                             nn.Linear(hidden_units, len(image_datasets['train'].classes)))

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # Move the model to GPU if available
    model.to(device)

    # iterating over epochs
    for epoch in range(epochs):

        # for accumulating metrics
        train_loss = 0.0
        valid_loss = 0.0
        accuracy = 0.0

        # Training phase
        model.train()

        # iterating over training batches
        for inputs, labels in dataloaders['train']:

            # moving inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device).type(torch.long)

            # forward pass
            outputs = model(inputs)

            # calculate the loss
            loss = criterion(outputs, labels)

            # zeroing out gradients (optimizer.zero_grad())
            optimizer.zero_grad()

            # backpropagation
            loss.backward()

            # optimization step
            optimizer.step()

            # accumulating training loss
            train_loss += loss.item()/len(labels)

        # Validation phase
        model.eval()

        # iterating over validation batches
        for inputs, labels in dataloaders['valid']:

            # moving inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device).type(torch.long)

            # excluding the forward pass from computation graph i.e. no node will have requires_grad = True
            with torch.no_grad():
                outputs = model(inputs)

            # calculating the loss
            loss = criterion(outputs, labels)

            # accumulating the validation loss
            valid_loss += loss.item()/len(labels)

            # generating prediction
            _, predicted = torch.max(outputs, dim=1)

            # accumulating accuracy
            accuracy += torch.sum((predicted == labels).type(torch.float)).item()/len(labels)

        # Print the training and validation loss, and accuracy for each epoch
        print(f"Epoch {epoch+1}/{epochs}: ",
              f"Train Loss: {train_loss/len(dataloaders['train']):.4f} ",
              f"Valid Loss: {valid_loss/len(dataloaders['valid']):.4f} ",
              f"Valid Accuracy: {accuracy/len(dataloaders['valid'])*100:.2f}%",
              sep="\n")
        print()

    # couple of print statements to make output interpretation easier
    print()
    print()

    # creating save_dir directory if it does not exist
    SAVE_DIR = Path(save_dir)
    if SAVE_DIR.is_dir():
        print(f"{save_dir} already present.")
        print()
    else:
        SAVE_DIR.mkdir(parents=True,
                       exist_ok=True)
        print(f"Created directory: {save_dir}")
        print()

    # Save the checkpoint
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'class_to_idx': image_datasets['train'].class_to_idx,
        'device_trained_on': device
    }
    torch.save(checkpoint, SAVE_DIR / Path('checkpoint.pth'))

    # save the model
    torch.save(model.state_dict(), SAVE_DIR / Path('saved_weights.pt'))

    # message that checkpoint and state dictionary are saved
    print(f"Checkpoint saved at: {SAVE_DIR / Path('checkpoint.pth')}, Model state dict saved at: {SAVE_DIR / Path('saved_weights.pt')}")

# using if __name__ == "__main__"
if __name__ == "__main__":
    # using argparse
    args = argument_parser()

    # calling training function
    train_model(args.arch,
                args.data_dir, 
                args.save_dir, 
                args.learning_rate,
                args.hidden_units,
                args.epochs,
                args.gpu)
    