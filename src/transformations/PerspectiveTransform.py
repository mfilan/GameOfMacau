from transformations.PerspectiveTransformHelper import PerspectiveTransformHelper


class PerspectiveTransform(PerspectiveTransformHelper):
    """Changes the perspective of image and its polygons"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, sample):
        image, card_polygon, label_polygons = sample['image'], sample['card_polygon'], sample['label_polygons']
        aug_image, projection_matrix = self.change_perspective(image, rotation=True)
        aug_card_polygon = self._rotate_polygon(card_polygon, projection_matrix)
        aug_label_polygons = [self._rotate_polygon(polygon, projection_matrix) for polygon in label_polygons]

        return {'image': aug_image, 'card_polygon': aug_card_polygon, 'label_polygons': aug_label_polygons}