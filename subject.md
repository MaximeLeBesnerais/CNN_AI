# CNN Implementation from Scratch

This project requires the implementation of an API for Convolutional Neural Network (CNN) model training and inference from scratch, without the use of high-level AI programming frameworks like PyTorch or TensorFlow.

## Requirements

The implementation must satisfy the following requirements:

* [cite_start]**No High-Level Frameworks**: Do not use AI programming frameworks such as PyTorch, Tensorflow, etc. [cite: 1]
* **Flexible Architecture**: Allow for flexible definition of the CNN architecture.
* [cite_start]**Activation Functions**: Support multiple activation functions, including ReLU and Leaky ReLU. [cite: 3]
* **Task Versatility**: Capable of both classification and regression tasks.
* **Weight Initialization**: Provide various options for weight initialization.
* **Optimizers**: Implement SGD with options for Momentum, RMSProp, and Adam.
* **Stopping Criteria**: Include criteria for stopping SGD.
* **Regularization**: Support L1, L2, and Elastic Net regularization.

## Implementation Details

### Layers

The following layers must be implemented:

* [cite_start]Conv2d [cite: 10]
* [cite_start]Pooling (MaxPooling, AvgPooling) [cite: 10]
* [cite_start]Drop-out [cite: 10]
* [cite_start]Batch Norm [cite: 10]
* [cite_start]Flatten [cite: 10]
* [cite_start]Fully Connected (FC) [cite: 10]

### Optimization

* Optimized convolution layer implementations such as `im2col`/`col2im` and FFT (Fast Fourier Transform) are required.

### Architecture Blocks

Support for the following architectural blocks is necessary:

* [cite_start]Inception Module [cite: 4]
* [cite_start]Residual Block [cite: 4]
* [cite_start]Depthwise conv/Bottleneck [cite: 4]

## Architectures

Implementation of established CNN architectures is recommended, based on the custom-built components.

* [cite_start]**Recommended Architectures**: FaceNet, MobileFaceNet, YOLO v4/5. [cite: 5]
* [cite_start]**Bonus**: A CNN + Transformer implementation. [cite: 6]

## Recommended Datasets

### For FaceNet/MobileFaceNet:

* [cite_start]**Labeled Faces in the Wild (LFW)**: For evaluating performance in unconstrained face recognition. [cite: 11]
* [cite_start]**MS-Celeb-1M**: Used to train MobileFaceNet for learning discriminative facial features. [cite: 12]
* [cite_start]**CASIA-WebFace**: A commonly used training dataset for MobileFaceNet. [cite: 13]

### For YOLO:

* [cite_start]**COCO (Common Objects in Context)**: A large-scale object detection dataset with 80 categories. [cite: 14]
* [cite_start]**Pascal VOC**: A classic object detection dataset with 20 object categories and over 11,000 images. [cite: 15, 16]
* [cite_start]**OpenImages**: A large dataset with over 1.7 million training images. [cite: 17]

## Notice

* [cite_start]Generated code is acceptable but must satisfy the above requirements. [cite: 7]
* [cite_start]You must be able to demonstrate a thorough understanding of the implementation. [cite: 8]
* [cite_start]You are required to run the implementation and provide detailed results, including the saved model file, a confusion matrix, and other evaluations. [cite: 9]
* [cite_start]A report on the key points of your design and implementation is required. [cite: 10]
* It is highly recommended to use programming languages other than Python and to explore parallel training with multi-threading or GPUs.