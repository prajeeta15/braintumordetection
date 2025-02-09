# braintumordetection

![image](https://github.com/prajeeta15/braintumordetection/assets/96904203/08290456-60a3-46ab-9989-28e97f0052bb)
![image](https://github.com/prajeeta15/braintumordetection/assets/96904203/70b00293-c553-4fcf-a98c-eb250ddfce83)
![image](https://github.com/prajeeta15/braintumordetection/assets/96904203/9090c91e-4e32-494b-b688-4165ec9e3a21)

packages used and why:

numpy :
NumPy is a fundamental package for numerical computing in Python. 

In brain detection, NumPy might be used to handle image data as arrays, perform array manipulations, and for preprocessing tasks such as normalization or data augmentation.

torch  :
PyTorch is a deep learning library that provides tools for building and training neural networks. 

PyTorch would be used to define the neural network architecture for brain detection, handle the training loop, compute gradients, and update model weights. It also provides utilities for loading datasets and performing common transformations.

glob :
glob module is used to find all the pathnames matching a specified pattern according to the rules used by the Unix shell.

In brain detection, glob can be used to load image files from a directory efficiently. For instance, you might use it to gather all MRI or CT scan images from a dataset folder.

matplotlib :
Matplotlib is a plotting library used for creating static, interactive, and animated visualizations in Python.

During the brain detection project, Matplotlib can be used to visualize the images, the results of the detection, and metrics such as training loss and accuracy over epochs. It helps in understanding how well the model is performing and in diagnosing issues.

random :
The random module implements pseudo-random number generators for various distributions.

In a brain detection project, random might be used for data augmentation techniques such as random cropping, rotations, or shuffling the dataset before training to ensure that the model generalizes well and does not overfit.

cv2 :
OpenCV is an open-source computer vision and machine learning software library. It provides a wide range of tools for image processing.

OpenCV can be used for preprocessing tasks such as resizing images, converting color spaces, and applying filters. It might also be used for post-processing the output of the neural network, such as drawing bounding boxes around detected brain tumors.

sys :
The sys module provides access to some variables used or maintained by the Python interpreter and to functions that interact strongly with the interpreter.

In a brain detection project, sys might be used to handle command-line arguments, allowing the user to specify parameters such as input data paths, model save paths, or hyperparameters for the training process.

Dataset taken from Kaggle : https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

Torch DataSet Class: An abstract class representing a Dataset. All other datasets should subclass it. All subclasses should override __len__, that provides the size of the dataset, and __getitem__ supporting integer indexing in range from 0 to len(self) exclusive.

Creating Dataloader: 
what is a dataloader?
DataLoader is a generic utility to be used as part of your application's data fetching layer to provide a simplified and consistent API over various remote data sources such as databases or web services via batching and caching.

In PyTorch, DataLoader is a built-in class that provides an efficient and flexible way to load data into a model for training or inference. It is particularly useful for handling large datasets that cannot fit into memory, as well as for performing data augmentation and preprocessing.

The DataLoader class works by creating an iterable dataset object and iterating over it in batches, which are then fed into the model for processing. The dataset object can be created from a variety of sources, including NumPy arrays, PyTorch tensors, and custom data sources such as CSV files or image directories.

Need to download Nvidia driver on system- http://www.nvidia.com/Download/index.aspx

What is Deep Learning?
Deep Learning, which has emerged as an effective tool for analyzing big data ,uses complex algorithms and artificial neural networks to train machines/computers so that they can learn from experience, classify and recognize data/images just like a human brain does

what is CNN in deep learning?
In Deep Learning, a Convolutional Neural Network or CNN is a type of artificial neural network, which is widely used for image/object recognition and classification. Deep Learning thus recognizes objects in an image by using a CNN.

The cnn architecture uses a special technique called Convolution instead of relying solely on matrix multiplications like traditional neural networks. Convolutional networks use convolution, which in mathematics is a mathematical operation on two functions that produces a third function that expresses how the shape of one is modified by the other.

How CNN Works
Input Image: An image is input into the network. For example, a grayscale image of size 28x28 pixels is represented as a 2D array of pixel values.
Convolutional Layer: Filters (e.g., 3x3 or 5x5) convolve over the image, detecting features such as edges, textures, and patterns.
Activation Function: A ReLU activation function is applied to the output of the convolution to introduce non-linearity.
Pooling Layer: Max pooling reduces the dimensionality, keeping the most significant features.
Stacking Layers: Multiple convolutional and pooling layers are stacked to learn complex features.
Flattening: The output from the convolutional and pooling layers is flattened into a single long vector.
Fully Connected Layer: This vector is fed into a fully connected layer to perform high-level reasoning.
Output Layer: The final output layer produces class scores, often passed through a softmax function for classification tasks.

Why use CNN for brain tumor detection?
- CNN can automatically learn and extract features from raw pixel data.
- they learn hierarchial features, starting from low=level features like edges and textures to high-level features which is crucial for complex patterns that are associated with tumors
- they explot the spatial strcuture of images by applying filters to small regions of the input which helps in detecting local features that are important for identifying tumors.
- ppoling layers in cnn help in achieving translation invariance, meaning the network can recognize tumors regardless of their position in the image.
- can handle large data efficiently.
- can process high resolution medical images like MRI and CT scans


