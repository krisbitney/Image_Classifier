import torch
from PIL import Image
import json
import argparse
import Utils


# parse command line
parser = argparse.ArgumentParser(description='Train neural networks for image classification')
parser.add_argument('image', type=str, action='store', help='image file path')
parser.add_argument('checkpoint', type=str, action='store', help='saved model checkpoint')
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu', help='use GPU for training')
parser.add_argument('--top_k', type=int, action='store', default=1, dest='topk', help='display prediction likelihood of top k classes')
parser.add_argument('--category_names', type=str, dest='category_names', help='filepath to json dictionary linking unique integer values to category names')
args = parser.parse_args()


# set computation device
device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')


# predict the class from an image file
def predict(image_path, model, classifier, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = Image.open(image_path)
    np_image = Utils.process_image(image)
    tensor_image = torch.tensor(np_image, dtype=torch.float32, device=device)
    tensor_image = torch.unsqueeze(tensor_image, 0)
    with torch.no_grad():
        classifier.eval()
        output = model(tensor_image)
        output = classifier(output)
        classifier.train()
    
    ps = torch.exp(output)
    probabilities, classes = ps.topk(topk, dim=1)
   
    probabilities = list(probabilities.data.cpu().numpy().squeeze())
    classes = list(classes.data.cpu().numpy().squeeze())
    
    class_names = [classifier.class_to_idx.get(str(cat), str(cat)) for cat in classes]
    
    return probabilities, class_names


# load classifier
classifier, model, _ = Utils.load_checkpoint(args.checkpoint)
model, classifier = model.to(device), classifier.to(device)

# add category names
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    classifier.class_to_idx = cat_to_name

# predict classes
probabilities, predictions = predict(args.image, model, classifier, args.topk)

# report results
print(f'Top {args.topk} predicted class probabilities:')
for p, k in zip(probabilities, predictions):
    print(f'{100*p:.2f}% - {k}')
