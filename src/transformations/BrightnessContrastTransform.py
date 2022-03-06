import cv2
import numpy as np


class BrightnessContrastTransform:
    """Changes the brightness and contrast of the image"""

    def __init__(self, normal_dist_brightness=(0, 40), contrast_dist_range=(0, 10)):
        self.normal_dist_brightness = normal_dist_brightness
        self.contrast_dist_range = contrast_dist_range

    def __call__(self, sample):
        image = sample['image'].copy()
        brightness = int(np.random.normal(*self.normal_dist_brightness))
        contrast = int(np.random.uniform(*self.contrast_dist_range))
        aug_image = self.apply_brightness_contrast(image, brightness, contrast)
        sample['image'] = aug_image

        return sample

    @staticmethod
    def apply_brightness_contrast(input_img, brightness=0, contrast=0):

        alpha_channel = input_img[:, :, 3].copy()
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
        buf[:, :, 3] = alpha_channel
        return buf