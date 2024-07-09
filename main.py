from image_loader import ImageLoader
from image_shower import ImageShower

from style_transfer import StyleTransferModel

from model_data import vgg19_model, resnet50_model, mobilenet_model

import matplotlib.pyplot as plt

import os

images_directory = 'images'

image_filenames = os.listdir(images_directory)

content_images = []
style_images = []

for image_file in image_filenames:
    if image_file.startswith('color'):
        style_images.append(image_file)
    else:
        content_images.append(image_file)

for n, c in enumerate(content_images):
    print(f'{n}: {c}')
print()

content_selected = int(input('Select a content image by index: '))
print()

for n, c in enumerate(style_images):
    print(f'{n}: {c}')
print()

style_selected = int(input('Select a style image by index: '))
print()

content_path = f'images/{content_images[content_selected]}'
style_path = f'images/{style_images[style_selected]}'

content_image = ImageLoader(content_path)
style_image = ImageLoader(style_path)

image_shower = ImageShower()

model = StyleTransferModel(model=vgg19_model, iterations=200)
best_image, best_loss, image_progress, metrics = model.apply_style_transfer(content_image, style_image)

# plotting results
fig, axs = plt.subplots(2, 2, figsize=(10,10))

axs[0, 0].plot(metrics['loss'])
axs[0, 0].set_title('Total Losses')

axs[0, 1].plot(metrics['psnr'])
axs[0, 1].set_title('PSNR')

axs[1, 0].plot(metrics['ssim'])
axs[1, 0].set_title('SSIM')

axs[1, 1].plot(metrics['mse'])
axs[1, 1].set_title('MSE')

plt.tight_layout()
plt.show()


image_shower.show_image_progress(image_progress)

image_shower.show_side_by_side(content_image, style_image)

plt.imshow(best_image)
plt.show()