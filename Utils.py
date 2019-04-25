import torch
from torchvision import models
import numpy as np
from Classifier import Classifier
import os


#Save the checkpoint
def save_checkpoint(classifier, filepath, model='resnet18', input_output=(1000, 102), loss=float('inf')):
    checkpoint = {'input_size': input_output[0],  # this comes from use of resnet
                  'output_size': input_output[1],  # this comes from use of iris flower dataset
                  'cl_hidden_layers': [hidden.out_features for hidden in classifier.hidden_layers],
                  'state_dict': classifier.state_dict(),
                  'min_validation_loss': loss,
                  'model_arch': model}
    torch.save(checkpoint, os.path.join(filepath, "checkpoint.pth"))
    
    
# loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # build classifier
    classifier = Classifier(checkpoint['input_size'],
                            checkpoint['output_size'],
                            checkpoint['cl_hidden_layers'])
    # load classifier state
    classifier.load_state_dict(checkpoint['state_dict'])
    # load the validation loss from the time the model was saved
    min_validation_loss = checkpoint['min_validation_loss']
    # load pre-trained model architecture
    model = get_model(checkpoint['model_arch'])
    return classifier, model, min_validation_loss


def get_model(arch='resnet18'):
    if arch == 'resnet18':
        model = models.resnet50(pretrained=True)
    elif arch == 'resnet34':
        model = models.resnet34(pretrained=True)
    else:
        model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # resize image
    width, height = image.size
    if width > height:
        new_width = int(256 * float(width)/float(height))
        image = image.resize((new_width, 256))
    else:
        new_height = int(256 * float(height)/float(width))
        image = image.resize((256, new_height))

    # crop image
    width, height = image.size
    w1 = (width - 224)/2
    w2 = w1 + 224
    h1 = (height - 224)/2
    h2 = h1 + 224
    image = image.crop((w1, h1, w2, h2 ))

    # convert pixels to numpy array, scale pixel values, and transpose
    np_image = np.array(image)/255
    np_image = np.transpose(np_image, axes=[2, 0, 1])

    # normalize image values
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]    
    for i in range(3):
        for j in range(224):
            np_image[i][j]= (np_image[i][j] - means[i]) / stds[i]
        
    return np_image
