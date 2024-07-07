import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from PIL import Image
import json
import argparse

def argument_parser():
    # creating ArgumentParser object
    parser = argparse.ArgumentParser(description='Train a deep learning model')

    # adding arguments
    parser.add_argument("--image_path",
                        type = str,
                        default = './flowers/test/1/image_06743.jpg',
                        help='Path to image file to do inference on.')

    parser.add_argument('--category_names',
                        type = str,
                        default = "./cat_to_name.json",
                        help='The mapping from folder labels to ground truth values.')

    parser.add_argument('--topk',
                        type = int,
                        default = 5,
                        help ='Top-K classes to display.')

    parser.add_argument('--checkpoint',
                        type = str,
                        default = "./saved_model/checkpoint.pth",
                        help ='The general checkpoint save path.')

    parser.add_argument('--model_path',
                        type = str,
                        default = "./saved_model/saved_weights.pt",
                        help ='The model state dictionary save path.')

    parser.add_argument('--gpu',
                        action="store_true",
                        help = 'Switch to GPU for optimal performance')

    # parsing arguments
    args = parser.parse_args()

    # returning args
    return args
    

def predict(image_path, checkpoint_path, model_path, category_names, use_gpu, topk):

    # setting up device
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

    # printing the device being used
    print(f"Using: {str(device)}")
    print()

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load the model architecture
    model = getattr(models, checkpoint['arch'])(pretrained=True)

    # changing the classifier head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, checkpoint["hidden_units"]),
                             nn.ReLU(),
                             nn.Linear(checkpoint["hidden_units"], 102))

    # taking different conditions into account (saved on CPU and loading on GPU (or vice-versa))
    if str(checkpoint['device_trained_on']) == "cuda:0" and str(device) == "cpu":
        map_location_current = device
        model.load_state_dict(torch.load(model_path, map_location=map_location_current))    # loading model weights
    elif str(checkpoint['device_trained_on']) == "cpu" and str(device) == "cuda:0":
        map_location_current = str(device)
        model.load_state_dict(torch.load(model_path, map_location=map_location_current))    # loading model weights
    else:
        # when saving and loading devices are the same (GPU-GPU or CPU-CPU)
        model.load_state_dict(torch.load(model_path))

    # Load the category to names mapping
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Loading the image using Pillow
    image = Image.open(image_path)

    # transforming the image and adding batch dimension
    preprocess = transforms.Compose([
        transforms.Resize(size=(256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)

    # Move the model and image tensor to device
    model.to(device)
    image = image.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation (or excluding from computation graph)
    with torch.no_grad():
        # Forward pass
        output = model(image)

    # computing probabilities and corresponding indices
    probabilities = torch.softmax(output, dim=1)
    top_probabilities, top_indices = probabilities[0].topk(topk)

    # Getting top classes
    idx_to_class = {v: k for k, v in checkpoint["class_to_idx"].items()}
    top_classes = [cat_to_name[idx_to_class[idx.item()]] for idx in top_indices]

    # Convert probabilities to numpy array
    top_probabilities = top_probabilities.tolist()

    # printing top-k classes and corresponding probabilities
    for pred_class, prob in zip(top_classes, top_probabilities):
        print(f"Predicted class: {pred_class}, Probability: {prob:.2f}") 

# executing this block only if running script directly
if __name__ == "__main__":

    # parsing command line arguments
    args = argument_parser()

    # predicting on the given image
    predict(args.image_path,
            args.checkpoint,
            args.model_path,
            args.category_names,
            args.gpu,
            args.topk)
