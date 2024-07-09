import numpy as np

import matplotlib.pyplot as plt

from image_loader import ImageLoader

class ImageShower:
    image_type = 'uint8'

    def _normalize_image_for_show(self, image_array):
        return np.squeeze(image_array, axis=0).astype(self.image_type)

    def show_image(self, image_array, title=None):
        normalized_image = self._normalize_image_for_show(image_array)

        if title:
            plt.title(title)

        plt.imshow(normalized_image)

    def show_side_by_side(self, content_image: ImageLoader, style_image: ImageLoader):
        content_image_array = content_image.get_image_array().astype(self.image_type)
        style_image_array = style_image.get_image_array().astype(self.image_type)

        plt.subplot(1, 2, 1)
        self.show_image(content_image_array, 'Content Image')

        plt.subplot(1, 2, 2)
        self.show_image(style_image_array, 'Style Image')
        
        plt.show()
    
    def show_combined_result(self, content_image: ImageLoader, style_image: ImageLoader, target_image_array):
        plt.figure(figsize=(10,10))
        self.show_side_by_side(content_image, style_image)
        self.show_image(target_image_array, 'Target Ouptut Image')
        plt.show()

    def show_image_progress(self, image_progress):
        plt.figure(figsize=(14,4))
        for i, image in enumerate(image_progress):
            plt.subplot(2, 5, i + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
        plt.show()