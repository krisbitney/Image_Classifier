# Image_Classifier
An educational deep learning project. The goal is to develop a generalizable neural network architecute capable of accuratley distinguishing image classes. The network is capable of identifying Iris flowers, among a dataset containing 102 flower categories, with at least 80% accuracy. For this project, I completed the following tasks:
1. Loaded and preprocessed image data
2. Created image classification neural network layer, and attached layer to transfer model
3. Created training function using an early-stopping decision rule
4. Evaluated neural network using test dataset
5. Created functions to save and load model
6. Predicted probabilities that images belong to each class

The model was trained using a negative log likelihood loss metric (i.e. maximum likelihood). Model testing performance was evaluated using accuracy.

This project uses PyTorch, Numpy, Pyplot, PIL, and json libraries.
