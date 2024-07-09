from dataclasses import dataclass

from typing import Any

from tensorflow.keras.applications import vgg19, resnet50, mobilenet

@dataclass
class ModelData:
    base_model: Any
    content_layers: list[str]
    style_layers: list[str]
    model_module: Any

vgg19_model = ModelData(
    base_model=vgg19.VGG19(include_top=False, weights='imagenet'), 
    content_layers=['block5_conv2'],
    style_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],
    model_module=vgg19
)

resnet50_model = ModelData(
    base_model=resnet50.ResNet50(include_top=False, weights='imagenet'), 
    content_layers=['conv4_block5_1_relu','conv5_block3_2_relu'],
    style_layers=['conv2_block1_2_relu',
                   'conv2_block2_1_relu', 
                   'conv3_block2_1_relu',
                   'conv3_block4_3_conv', 
                   'conv4_block4_2_relu',
                   'conv4_block6_2_relu',
                   'conv5_block2_2_relu', 
                   'conv5_block3_2_relu'],
    model_module=resnet50
)

mobilenet_model = ModelData(
    base_model=mobilenet.MobileNet(include_top=False, weights='imagenet'), 
    content_layers=['conv_pw_12'],
    style_layers=['conv_pw_1',
                   'conv_pw_2', 
                   'conv_pw_3',
                   'conv_pw_4'],
    model_module=mobilenet
)