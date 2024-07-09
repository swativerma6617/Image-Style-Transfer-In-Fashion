from PIL import Image

import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array

class ImageLoader:
    max_dimension = 512
    expected_clipped_image_size = 4
    image_type = 'uint8'

    normalized_means = np.array([103.939, 116.779, 123.68])

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = self._resize()
    
    def _resize(self) -> Image:
        original_image = Image.open(self.image_path)
        original_x, original_y = original_image.size

        scale_factor = self.max_dimension / max(original_x, original_y)
        new_x, new_y = round(original_x * scale_factor), round(original_y * scale_factor)
        
        return original_image.resize((new_x, new_y), Image.Resampling.LANCZOS)
    
    def get_bw_image_array(self):
        bw_image = self.image.convert('L')
        image_array = img_to_array(bw_image)
        return np.expand_dims(image_array, axis=0)

    def get_image_array(self):
        image_array = img_to_array(self.image)
        return np.expand_dims(image_array, axis=0)
    
    def process_image_for_model(self, model_module):
        image_array = self.get_image_array()
        return model_module.preprocess_input(image_array)
    
    # this might have to be a utility function
    def clip_image_from_model(self, processed_image):
        clipped_image = processed_image.copy()
        clipped_image = np.squeeze(clipped_image, 0)
        clipped_image = self._inverse_image_process(clipped_image)
        return np.clip(clipped_image, 0, 255).astype(self.image_type)

    def _inverse_image_process(self, processed_image):
        # removing zero center by mean pixel 
        processed_image[:, :, 0] += self.normalized_means[0]
        processed_image[:, :, 1] += self.normalized_means[1]
        processed_image[:, :, 2] += self.normalized_means[2]
        
        # correct ordering of RGB
        return processed_image[:, :, ::-1]