# Efficient implementation of Haar-like features using Convolution
This repository implements Haar-Like features using convolutions in PyTorch. 

Within the repository, implementation is provided for the following:
- **2D Haar-Like features** for Grayscale images following method from: Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Proceedings of the 2001 IEEE computer society conference on computer vision and pattern recognition. CVPR 2001. Vol. 1. Ieee, 2001.
![haar3d](https://raw.githubusercontent.com/masadcv/PyTorchHaarFeatures/master/data/diagrams-Haar2D.png)


- **3D Haar-Like features** for 3D image data, e.g. medical images, following method from: Jung, Florian, Matthias Kirschner, and Stefan Wesarg. "A generic approach to organ detection using 3d haar-like features." Bildverarbeitung für die Medizin 2013. Springer, Berlin, Heidelberg, 2013. 320-325.
![haar3d](https://raw.githubusercontent.com/masadcv/PyTorchHaarFeatures/master/data/diagrams-HaarHor3D.png)

Both 2D and 3D versions of Haar-Like features have been implemented using convolutions in PyTorch and hence can be embedded into a given network where hand-crafted Haar-Like features are needed.

Reference code from https://github.com/rohitghosh/3DViolaJones helped us in initial understanding of Haar-Like features, which led to our implementation using PyTorch, where a few more features were added for 3d Haar-Like case. 

## Installation
This package can be installed as: 

`pip install torchhaarfeatures`

or 

`pip install git+https://github.com/masadcv/PyTorchHaarFeatures`

## Examples
Usage examples are provided in example python files within the repository.

A simple example (`example.py`) usage following a PyTorch usage format:

```
import torchhaarfeatures
import torch

haarfeat3d = torchhaarfeatures.HaarFeatures3d(kernel_size=(9, 9, 9), stride=1)
output_haar3d = haarfeat3d(torch.rand(size=(1, 1, 128, 128, 128)))

print(output_haar3d.shape)

haarfeat2d = torchhaarfeatures.HaarFeatures2d(kernel_size=(9, 9), stride=1)
output_haar2d = haarfeat2d(torch.rand(size=(1, 1, 128, 128)))
print(output_haar2d.shape)
```

More advanced usage embedded Haar-Like layers (`example2d_learning.py`) are feature extractor:
```
class MyCatSegnentorHaarlike(nn.Module):
    def __init__(
        self,
        kernel_size=6,
        hidden_layers=[32, 16],
        num_classes=2,
        haar_padding="same",
    ):
        super().__init__()
        self.haarfeatureextactor = torchhaarfeatures.HaarFeatures2d(
            kernel_size=kernel_size,
            padding=haar_padding,
            stride=1,
            padding_mode="zeros",
        )
        in_channels_current_layer = self.haarfeatureextactor.out_channels
        
        self.classifier = []
        for hlayer in hidden_layers:
            self.classifier.append(
                nn.Sequential(
                    *[
                        nn.Conv2d(
                            in_channels=in_channels_current_layer,
                            out_channels=hlayer,
                            kernel_size=1,
                            stride=1,
                            padding="same",
                        ),
                        nn.ReLU(),
                        nn.Dropout2d(p=0.3),
                    ]
                )
            )
            in_channels_current_layer = hlayer

        # add final layer
        self.classifier.append(
            nn.Conv2d(
                in_channels=in_channels_current_layer,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
            )
        )
        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, x):
        x = self.haarfeatureextactor(x)
        x = self.classifier(x)
        return x   
```

![image](https://raw.githubusercontent.com/masadcv/PyTorchHaarFeatures/master/data/example2d_learning_figure_1.png)

## Citation
If you use our code, please consider citing our paper:

```
Asad, Muhammad, Lucas Fidon, and Tom Vercauteren. 
"ECONet: Efficient Convolutional Online Likelihood Network for Scribble-based Interactive Segmentation." 
arXiv preprint arXiv:2201.04584 (2022).
```
