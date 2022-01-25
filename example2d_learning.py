import pt_haarfeatures

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
import matplotlib.pyplot as plt


class MyCatSegnentorHaarlike(nn.Module):
    def __init__(
        self,
        kernel_size=6,
        hidden_layers=[32, 16],
        num_classes=2,
        haar_padding="same",
    ):
        super().__init__()
        self.haarfeatureextactor = pt_haarfeatures.HaarFeatures2d(
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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device {}".format(device))

    model = MyCatSegnentorHaarlike().to(device)
    
    input_image_np = np.asarray(Image.open("data/cat.76.jpg").convert("L")).astype(np.float32)
    output_mask_np = (np.asarray(Image.open("data/mask_cat.76.jpg"))>0).astype(np.long)

    print(input_image_np.shape)
    print(input_image_np.dtype)

    print(output_mask_np.shape)
    print(output_mask_np.dtype)
    print(np.unique(output_mask_np))

    input_image_np = np.expand_dims(np.expand_dims(input_image_np, axis=0), axis=0)
    output_mask_np = np.expand_dims(output_mask_np, axis=0)

    input_tensor = torch.from_numpy(input_image_np).to(device)/255
    output_tensor = torch.from_numpy(output_mask_np).to(device)

    print(input_tensor.type)
    print(output_tensor.type)

    # train all except haar featureextractor
    params_to_train = [
                p for n, p in model.named_parameters() if "featureextactor" not in n
            ]
    optim = torch.optim.Adam(params_to_train, lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    nepochs = 1000
    for i in range(nepochs):
        out = model(input_tensor)
        loss = loss_func(out, output_tensor)
        loss.backward()
        optim.step()
        optim.zero_grad()

        print(loss)

    netout_np = out.detach().cpu().squeeze().numpy()

    print(netout_np.shape)

    netout_np = np.argmax(netout_np, axis=0) * 255

    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(input_image_np.astype(np.uint8)), cmap="gray")

    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(output_mask_np.astype(np.uint8) * 255))

    plt.subplot(1, 3, 3)
    plt.imshow(netout_np.astype(np.uint8))
    plt.suptitle("Training output")
    plt.show()
    plt.close()
    
    input_image_np = np.asarray(Image.open("data/test-tabby-edited.jpg").convert("L")).astype(np.float32)
    input_image_np = np.expand_dims(np.expand_dims(input_image_np, axis=0), axis=0)
    input_tensor = torch.from_numpy(input_image_np).to(device)/255

    model.eval()

    out = model(input_tensor)
    netout_np = out.detach().cpu().squeeze().numpy()

    netout_np = np.argmax(netout_np, axis=0) * 255

    plt.subplot(1, 2, 1)
    plt.imshow(np.squeeze(input_image_np.astype(np.uint8)), cmap="gray")

    plt.subplot(1, 2, 2)
    plt.imshow(netout_np.astype(np.uint8))

    plt.suptitle("Testing output")
    plt.show()
