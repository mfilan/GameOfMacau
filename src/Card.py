import json
import os
from pathlib import Path

import cv2


class Card:
    def __init__(self, name):
        self.name = name
        self.image = None
        self.card_polygon = None
        self.label_polygons = []

    def set_image(self, image):
        self.image = image

    def set_card_polygon(self, card_polygon):
        self.card_polygon = card_polygon

    def set_label_polygons(self, label_polygons):
        self.label_polygons = label_polygons

    def add_label_polygon(self, label_polygon):
        self.label_polygons.append(label_polygon)

    def save(self, dir_name='../interim_data'):
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        template = {"filename": self.name,
                    "card_polygon": self.card_polygon,
                    "label_polygons": self.label_polygons}

        json_name = os.path.join(dir_name, self.name + '.json')
        with open(json_name, 'w') as fp:
            json.dump(template, fp)
        print(f"{json_name} saved correctly!")

        image_name = os.path.join(dir_name, self.name + '.png')
        cv2.imwrite(image_name, self.image)
        print(f"{image_name} saved correctly!")