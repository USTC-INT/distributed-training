import torch
import torchvision.models as models
import numpy as np

test_models = [
    models.resnet18(),
    models.resnet50(),
    models.alexnet(),
    models.vgg16(),
    models.densenet161(),
    models.inception_v3()
]

for m in test_models:
    layer_size = []
    for name,layer in m.named_parameters():
        para_array = torch.nn.utils.parameters_to_vector(layer)
        layer_size.append(para_array.nelement() * para_array.element_size() / 1024 / 1024)

    print("Model size: {} MB".format(sum(layer_size)))
    # print("Per layer size (MB): {}".format(layer_size))
