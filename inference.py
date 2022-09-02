import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import argparse
from pathlib import Path
import torch
import json

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model', metavar='MODEL', default = 'resnet18',
                    help='model architecture default: resnet18')
parser.add_argument('--image',
                    help='test image url')

def main():
    args = parser.parse_args()

    model = timm.create_model(args.model, pretrained=True)
    model.eval()

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    url, filename = (args.image, Path(args.image).name)
    urllib.request.urlretrieve(url, filename)
    img = Image.open(filename).convert('RGB')
    tensor = transform(img).unsqueeze(0) # transform and add batch dimension

    with torch.no_grad():
        out = model(tensor)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    
    # Get imagenet class mappings
    url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    urllib.request.urlretrieve(url, filename) 
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Print top category per image
    top_prob, top_catid = torch.topk(probabilities, 1)
    for i in range(top_prob.size(0)):
        print('{' + f'\"predicted\" : \"{str(categories[top_catid[i]])}\", \"confidence\" : \"{str(round(float(top_prob[i].item()), 2))}\"' + '}')
    
if __name__ == '__main__':
    main()