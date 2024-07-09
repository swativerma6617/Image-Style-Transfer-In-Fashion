import cv2

from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse

import numpy as np

class MetricsTracker:
    pixel_max = 255.0

    def __init__(self):
        self.loss_tracker = []
        self.psnr_tracker = []
        self.ssim_tracker = []
        self.mse_tracker = []

        self.best_image = None
        self.best_loss = float('inf')
    
    def _standardize_image(self, image):
        return cv2.cvtColor(image[0], cv2.COLOR_BGR2GRAY)
    
    def _calc_mse(self, base_image, current_image):
        base_image_std = self._standardize_image(base_image)
        current_image_std = self._standardize_image(current_image)

        return mse(base_image_std, current_image_std)

    def _calc_ssim(self, base_image, current_image):
        base_image_std = self._standardize_image(base_image)
        current_image_std = self._standardize_image(current_image)

        return ssim(base_image_std, current_image_std, data_range=current_image_std.max() - current_image_std.min())
    
    def _calc_psnr(self, base_image, current_image):
        mse = np.mean((base_image - current_image) ** 2)
        if mse == 0:
            return 100
        
        return 20 * np.log10(self.pixel_max / np.sqrt(mse))

    def track_best_loss_and_image(self, total_loss, content_image):
        self.loss_tracker.append(total_loss)
        if total_loss < self.best_loss:
            self.best_loss = total_loss 
            self.best_image = content_image
    
    def track_metrics(self, base_image, current_image):
        self.psnr_tracker.append(self._calc_psnr(base_image, current_image))
        self.ssim_tracker.append(self._calc_ssim(base_image, current_image))
        self.mse_tracker.append(self._calc_mse(base_image, current_image))

    def get_metrics(self):
        return {
            'loss': self.loss_tracker,
            'psnr': self.psnr_tracker,
            'ssim': self.ssim_tracker,
            'mse': self.mse_tracker
        }