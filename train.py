import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn, optim
import argparse
import os
from Classifier import Classifier
import Utils


# parse command line
parser = argparse.ArgumentParser(description='Train neural networks for image classification')
parser.add_argument('data_dir', type=str, action='store', help='training data directory')
parser.add_argument('--val_dir', type=str, action='store', dest='val_dir', help='validation data directory')
parser.add_argument('--save_dir', type=str, action='store', default=os.getcwd(), dest='save_dir', help='directory to save checkpoints (defaults to current working directory)')
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu', help='use GPU for training')

parser.add_argument('--arch', type=str, action='store', default='resnet18', dest='arch', help='pre-trained model architecture; supports resnet18 (default), resnet34, and resnet50')
parser.add_argument('--learn_rate', type=float, action='store', default=0.001, dest='learn_rate', help='set optimizer learning rate')
parser.add_argument('--hidden_units', type=int, action='append', default=[], dest='layers', help='adds dense layer with specified output')
parser.add_argument('--classes', type=int, action='store', default=102, dest='output_size', help='number of categories to classify')
parser.add_argument('--epochs', type=int, action='store', default=10, dest='epochs', help='number of training epochs')
parser.add_argument('--early_stop', action='store_true', default=False, dest='early_stop', help='keep checkpoint with minimum validation loss')

args = parser.parse_args()

# set computation device
device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

Print(f'Training {args.arch}-based neural network with feedforward classifier containing {zip(*args.layers)} hidden units for {args.epochs} epochs to predict {args.classes} classes. Computations will be completed using {device}. Adam optimization will use learning rate {args.learn_rate}.')
      
# Define your transforms for the training, validation, and testing sets
def process_data(train_dir, valid_dir=None, batch_size=100):
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(0.3),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                                           ])
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    # Using the image datasets and the trainforms, define the dataloader
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    if valid_dir:
        test_transform = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])
                                               ])
        valid_data = datasets.ImageFolder(valid_dir, transform=test_transform)
        validloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    else:
        validloader = None

    return trainloader, validloader


# create optimizer
def get_optim(classifier, learn_rate=0.001):
    return optim.Adam(classifier.parameters(), lr=learn_rate)


# Train model
def train_model(trainloader, model, classifier, optimizer, criterion, epochs=10, batch_size=100, save_filepath='', early_stop=False, validloader=None, min_validation_loss=float('inf')):
    classifier.train()
    stagnance = 0
    
    train_loss_history = []
    valid_loss_history = []
    valid_acc_history = []

    for e in range(1, epochs+1):

        # Performance tracking variables
        train_loss = 0
        valid_loss = 0
        valid_correct = 0

        # Training pass
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model(images)
            output = classifier(output)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # Save batch loss
            train_loss += loss.to(device).item()

        # Valiation pass
        else:
            if validloader:
                classifier.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)

                        output = model(images)
                        output = classifier(output)
                        loss = criterion(output, labels)

                        # Save batch loss
                        valid_loss += loss.item()

                        # Save count of accurate predictions
                        ps = torch.exp(output)
                        _, preds = ps.topk(1, dim=1)
                        valid_correct += torch.sum(preds.view(-1,) == labels).item()
                classifier.train()

        # Use batch mean loss statistics to account for difference in data set sizes
        mean_train_loss = train_loss/len(trainloader)
        if validloader:
            mean_valid_loss = valid_loss/len(validloader)
            valid_acc = 100 * valid_correct / (len(validloader) * batch_size)
        else:
            mean_valid_loss = 0
            valid_acc = 0

        # Save performance histories
        train_loss_history.append(mean_train_loss)
        valid_loss_history.append(mean_valid_loss)
        valid_acc_history.append(valid_acc)

        # Print performance statistics
        print(f'Epoch: {e} --  Mean Training Loss={mean_train_loss:.3f}.. '
              f'Mean Validation Loss = {mean_valid_loss:.3f}.. '
              f'Validation Accuracy = {valid_acc:.2f}%..')

        # Implement early stopping by saving classifier with minimum mean validation loss
        if early_stop:
            if mean_valid_loss < min_validation_loss:
                stagnance = 0
                min_validation_loss = mean_valid_loss
                print('Saving classifier...')
                Utils.save_checkpoint(classifier, save_filepath, model=args.arch, input_output=(1000, classifier.output_layer.out_features), loss=min_validation_loss)
            else:
                stagnance += 1
                if stagnance == 5:
                    print("No improvement in validation loss after 5 consecutive epochs. Stopping early...")
                    break
      
    histories = {'train_loss': train_loss_history, 
                 'valid_loss': valid_loss_history, 
                 'valid_accuracy': valid_acc_history}

    if not early_stop:
        print('Saving classifier...')
        Utils.save_checkpoint(classifier, save_filepath, model=args.arch, input_output=(1000, classifier.output_layer.out_features))

    return histories


# process data
batch_size = 100
trainloader, validloader = process_data(args.data_dir, valid_dir=args.val_dir, batch_size=batch_size)

# build model
model = Utils.get_model(args.arch)
layers = args.layers 
if not args.layers:
    layers = [700, 400]
classifier = Classifier(1000, args.output_size, layers)
optimizer = get_optim(classifier, learn_rate=args.learn_rate)
criterion = nn.NLLLoss()
model, classifier = model.to(device), classifier.to(device)

# train model
train_model(trainloader, model, classifier, optimizer, criterion,
            epochs=args.epochs, batch_size=batch_size, save_filepath=args.save_dir, validloader=validloader)
