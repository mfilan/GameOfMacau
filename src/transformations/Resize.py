import copy

from utils import *


class Resize:
    """Resizes image and polygons to given size relative to the longest side of image"""

    def __init__(self, size_range=(400, 600)):
        self.size_range = size_range

    def __call__(self, sample, size, on_canvas=False):
        if not on_canvas:
            image, card_polygon, label_polygons = sample['image'].copy(), sample['card_polygon'], sample[
                'label_polygons']
            aug_image, matrix = self.resize_image(image, size)
            aug_label_polygons = [rotate_polygon(polygon, matrix) for polygon in label_polygons]
            aug_card_polygon = rotate_polygon(card_polygon, matrix)
            return {'image': aug_image, "card_polygon": aug_card_polygon, 'label_polygons': aug_label_polygons}
        else:
            image, image_dict = sample['image'].copy(), copy.deepcopy(sample['image_dict'])
            aug_image, matrix = self.resize_image(image, size)
            for card in image_dict:
                label_polygons = image_dict[card]['label_polygons'].copy()
                aug_label_polygons = [rotate_polygon(polygon, matrix) for polygon in label_polygons]
                image_dict[card]['label_polygons'] = aug_label_polygons
            return {'image': aug_image, 'image_dict': image_dict}

    def get_random_size(self):
        return np.random.randint(*self.size_range)

    @staticmethod
    def resize_image(image, size):
        height, width = image.shape[:2]
        if height >= width:
            scaling_factor = size / height
            new_width = int(scaling_factor * width)
            matrix = np.float32([[scaling_factor, 0, 0],
                                 [0, scaling_factor, 0]])
            t = np.float32(matrix)
            aug_image = cv2.warpAffine(image, t, (new_width, size))
        else:
            scaling_factor = size / width
            new_height = int(scaling_factor * height)
            matrix = np.float32([[scaling_factor, 0, 0],
                                 [0, scaling_factor, 0]])
            t = np.float32(matrix)
            aug_image = cv2.warpAffine(image, t, (size, new_height))
        t = np.vstack((t, np.asarray([[0, 0, 1]])))
        return aug_image, t