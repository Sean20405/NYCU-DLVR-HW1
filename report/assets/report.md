# DLVR HW1 Report
## Intrduction
The task is to classify creature images into 100 different categories using ResNet. The core idea of my work is to leverage advanced data augmentation techniques, MixUp and CutMix and ensemble learning to improve the model's performance. And ResNeXt is chosen to be the backbone of the model.

### ResNet
ResNet (Residual Network) is a deep learning architecture introduced by Microsoft in 2015 to address the vanishing gradient problem in very deep neural networks. The main contribution is its residual block. It try to learn the residual, or difference, between layers by skip connections, which bypass one or more layers. The method allow the network to learn identity mappings.

### ResNeXt
ResNeXt is an improved variant of ResNet, introduced by Facebook AI in 2017, which enhances performance through seperate residual block into many small branch. Instead of simply increasing depth or width, ResNeXt introduces the concept of "cardinality," which refers to the number of parallel convolutional paths in each residual block. This design improves feature learning while maintaining computational efficiency. 
![image](https://hackmd.io/_uploads/rJByx6kaye.png)


### Mixup-Cutmix
Mixup and CutMix are data augmentation techniques that improve model generalization. Mixup blends two images and labels through linear interpolation, while CutMix replaces a region of one image with a patch from another and mix the labels. Both methods enhance robustness, reduce overfitting, and improve classification accuracy by promoting diverse feature learning.
![MixUp_CutMix](https://hackmd.io/_uploads/SySqDT1a1x.png)

### K-Fold Cross-Validation & Ensemble
K-Fold Cross-Validation splits data into K subsets, training K models with different (K-1) subsets as training set and 1 subset as validation sets. When testing, it combines those K models by averaging their output and predict an overall label.


## Method
### Data Pre-processing
Data preprocessing plays a crucial role in the training pipeline. Additional augmentations such as random horizontal flip, random rotation are applied before training. Random resize crop is also added to both increase robustness and fit model input requirement. Then, randomly choose mixUp or cutMix as the additional data augmentation techniques to improve the diversity of training dataset as well as smooth the labels. Note that there are no data augmentations in validation or testing. The only preprocessing in these period is resizing to 224x224.

### Model Architecture
The model is built upon the ResNeXt101 64x4d architecture, which has been pretrained on ImageNet and the weight is retrieved from PyTorch package as `ResNeXt101_64X4D_Weights.IMAGENET1K_V1`. Also, to adapt the model to this task, the final fully connected layer is modified to output predictions for 100 classes.

### Optimization
For optimization, AdamW is used with an initial learning rate of 1e-4. A cosine annealing warm restart scheduler dynamically adjusts the learning rate throughout training. Cross-entropy loss is employed as the loss function, and training is performed with a batch size of 64 over 80 epochs. Checkpoints are periodically saved, ensuring that the best-performing model based on validation accuracy is retained.

### Cross Validation and Test-Time Ensemble
To further enhance model performance, five-fold cross-validation is employed. This technique helps in reducing overfitting by training on different subsets of data. After training those 5 sub-models, a test-time ensemble is introduces. It averages the outputs of 5 sub-models and choose the index with maximum value as the predicted label.

### Hyperparameters
The detailed hyperparameters are listed below:

- Learning rate: 1e-4
- Optimizer: AdamW
- Scheduler: CosineAnnealingWarmRestarts
- Loss Function: CrossEntropyLoss
- Batch Size: 64
- Epochs: 80
- Number of Folds for Cross Validation: 5

## Result
I compare 


## References

## Supplementary