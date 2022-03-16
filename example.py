import torchhaarfeatures
import torch

haarfeat3d = torchhaarfeatures.HaarFeatures3d(kernel_size=(9, 9, 9), stride=1)
output_haar3d = haarfeat3d(torch.rand(size=(1, 1, 128, 128, 128)))

print(output_haar3d.shape)

haarfeat2d = torchhaarfeatures.HaarFeatures2d(kernel_size=(9, 9), stride=1)
output_haar2d = haarfeat2d(torch.rand(size=(1, 1, 128, 128)))
print(output_haar2d.shape)