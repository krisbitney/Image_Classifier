# Image_Classifier
An educational deep learning project. The goal is to develop a generalizable neural network architecute capable of accuratley distinguishing image classes. The network is capable of identifying Iris flowers, among a dataset containing 102 flower categories, with at least 80% accuracy. 

## For the first part of this project, I used deep learning methods to create an image classifier in a Jupyter notebook. See "Image Classifier Project.ipynb". 

I completed the following tasks:
1. Loaded and preprocessed image data
2. Created image classification neural network layer, and attached layer to transfer model
3. Created training function using an early-stopping decision rule
4. Evaluated neural network using test dataset
5. Created functions to save and load model
6. Predicted probabilities that images belong to each flower class

The model was trained using a negative log likelihood loss metric (i.e. maximum likelihood). Model testing performance was evaluated using accuracy.

## For the second part of this project, I generalized the neural network classification model for use with any image dataset and implemented its functionality in two command line applications. 

#### First application: train.py
The first application creates and trains an image classification neural network and includes options to adjust the network architecture and hyperparameters. To open the application, run "train.py" in the terminal.

```
usage: train.py [-h] [--val_dir VAL_DIR] [--save_dir SAVE_DIR] [--gpu]
                [--arch ARCH] [--learn_rate LEARN_RATE]
                [--hidden_units LAYERS] [--classes OUTPUT_SIZE]
                [--epochs EPOCHS] [--early_stop]
                data_dir

Train neural networks for image classification

positional arguments:
  data_dir              training data directory

optional arguments:
  -h, --help            show this help message and exit
  --val_dir VAL_DIR     validation data directory
  --save_dir SAVE_DIR   directory to save checkpoints (defaults to current
                        working directory)
  --gpu                 use GPU for training
  --arch ARCH           pre-trained model architecture; supports resnet18
                        (default), resnet34, and resnet50
  --learn_rate LEARN_RATE
                        set optimizer learning rate
  --hidden_units LAYERS
                        adds dense layer with specified output
  --classes OUTPUT_SIZE
                        number of categories to classify
  --epochs EPOCHS       number of training epochs
  --early_stop          keep checkpoint with minimum validation loss
```

#### Second application: predict.py
The second application predicts image class probabilities using a trained neural network that was created and trained with "train.py". To open the application, "predict.py" in the terminal.

```
usage: predict.py [-h] [--gpu] [--top_k TOPK]
                  [--category_names CATEGORY_NAMES]
                  image checkpoint

Train neural networks for image classification

positional arguments:
  image                 image file path
  checkpoint            saved model checkpoint

optional arguments:
  -h, --help            show this help message and exit
  --gpu                 use GPU for training
  --top_k TOPK          display prediction likelihood of top k classes
  --category_names CATEGORY_NAMES
                        filepath to json dictionary linking unique integer
                        values to category names
```
                     
 ## Required libraries                       
This project uses PyTorch, Numpy, Pyplot, PIL, argparse, Jupyter, and common Python built-in libraries.


