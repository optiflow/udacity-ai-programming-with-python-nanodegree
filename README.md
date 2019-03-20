# Coursework for Udacity's AI Programming with Python Nanodegree Program
---
Implementation of transfer learning with pre-trained Convolutional Neural Network (CNN) to develop flowers image classifier with PyTorch.

## Model Output
---
The model will return the top 5 predicted flower type based on input image.
<a href="https://drive.google.com/uc?export=view&id=1advdGeZ2WRD0Nd-agme71p-GdDaqPt_f"><img src="https://drive.google.com/uc?export=view&id=1advdGeZ2WRD0Nd-agme71p-GdDaqPt_f" style="width: 250px; max-width: 100%; height: auto" title="Click for the larger version." /></a>


## Building Custom Image Classification
---
- **train.py** contains the necessary codes to import pre-trained CNN, either VGG19 or Alexnet, and replace last Fully Connected Layer (FCL) to train custom image classification. 
- User should provide own images for customised image classification and update the folder path in **train.py**. 
- Ground truth of image should be provided in JSON format.



## License
---
The contents of this repository are covered under the [MIT License](https://opensource.org/licenses/MIT).
