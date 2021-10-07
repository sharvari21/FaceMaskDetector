# FaceMaskDetector
This model is build with OpenCV, TensorFlow, Keras, and MobileNet. This model can be used as real time application which will detect the face mask and will showing the result, for safety purposes of Covid-19.

## Requirements:

TensorFlow>=1.15.2
keras==2.3.1
imutils==0.5.3
numpy==1.18.2
opencv-python==4.2.0.*
matplotlib==3.2.1
scipy==1.4.1

## Datasets:

Basically image data used with two category one is with mask face images and another is without mask face images, it can be easily available on Kaggle.

## Model build with: 

### MobileNetV2: 
MobileNet-v2 is a convolutional neural network that is 53 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.
MobileNetV2 is very similar to the original MobileNet, except that it uses inverted residual blocks with bottlenecking features. It has a drastically lower parameter count than the original MobileNet. MobileNets support any input size greater than 32 x 32, with larger image sizes offering better performance.

### OpenCV’s blobFromImage:
This function performs Mean subtraction which is used to help combat illumination changes in the input images in our dataset. We can therefore view mean subtraction as a technique used to aid our Convolutional Neural Networks.
Before we even begin training our deep neural network, we first compute the average pixel intensity across all images in the training set for each of the Red, Green, and Blue channels.
This implies that we end up with three variables:
 ,  , and  
Typically, the resulting values are a 3-tuple consisting of the mean of the Red, Green, and Blue channels, respectively.
However, in some cases the mean Red, Green, and Blue values may be computed channel-wise rather than pixel-wise, resulting in an MxN matrix. In this case the MxN matrix for each channel is then subtracted from the input image during training/testing.
When we are ready to pass an image through our network (whether for training or testing), we subtract the mean,  , from each input channel of the input image:
 
 
 
We may also have a scaling factor,  , which adds in a normalization:
 
 
 
The value of   may be the standard deviation across the training set (thereby turning the preprocessing step into a standard score/z-score). However,  may also be manually set (versus calculated) to scale the input image space into a particular range — it really depends on the architecture, how the network was trained, and the techniques the implementing author is familiar with.
It’s important to note that not all deep learning architectures perform mean subtraction and scaling! Before you preprocess your images, be sure to read the relevant publication/documentation for the deep neural network you are using.

## Future Utility:
This project can be integrated with embedded systems for application in airports, railway stations, offices, schools, and public places to ensure that public safety guidelines are followed. With further improvements these types of models could be integrated with CCTV cameras to detect and identify people without masks.

### References:
1.	Video tutorial by Balaji Srinivasan.
2.	OpenCV’s blobFromImage reference from pyimagesearch.com
3.	MobileNetV2 reference from paperwithcode.com
4.	Opencv tutorial from edureka.com


