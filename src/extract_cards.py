import glob
import json

import numpy.lib.stride_tricks
from PIL import ImageDraw
from shapely.geometry import LineString

from Card import Card
from utils import *


def rolling_window(a, window_size):
    shape = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def get_longest_lines(card):
    """return lines [x1,y1,x2,y2]"""
    pairs_of_points = rolling_window(card.card_polygon, 2)
    points1, points2 = pairs_of_points[:, 0, :], pairs_of_points[:, 1, :]
    distances = np.apply_along_axis(np.linalg.norm, 1, points1 - points2)
    return pairs_of_points[distances.argsort()][-4:].reshape(-1, 4)


def extend_lines(lines, height, width):
    """extends line segments to border of image"""
    extended_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        numerator = (y2 - y1)
        denominator = (x2 - x1)
        if denominator == 0:
            a = numerator / 0.001
        else:
            a = numerator / denominator
        b = y1 - a * x1

        if abs(a) > 70:
            extended_lines.append([(-b) / a, 0, (height - b) / a, height])
        else:
            extended_lines.append([0, b, width, a * width + b])
    extended_lines = np.array(extended_lines).astype(np.int32)
    return extended_lines


def get_intersection_points(lines):
    lines = np.vstack((lines, np.expand_dims(lines[0], axis=0)))
    pairs_of_lines = rolling_window(lines, 2)
    intersections = []
    for pair in pairs_of_lines:
        a, b, c, d = pair.reshape(4, 2)
        line1 = LineString([a, b])
        line2 = LineString([c, d])

        int_pt = line1.intersection(line2)
        point_of_intersection = int_pt.x, int_pt.y
        intersections.append(point_of_intersection)
    return np.array(intersections).astype(np.float32)


def get_bbox(card):
    lines = get_longest_lines(card)
    height, width = card.image.shape[:2]
    extended_lines = extend_lines(lines, height, width)
    intersection_points = get_intersection_points(extended_lines[[0, 2, 1, 3]])
    return extended_lines, intersection_points


def get_size(pt1, pt2):
    return np.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))


def four_point_transform(image, pts):
    (tl, tr, br, bl) = pts
    width_a = get_size(br, bl)
    width_b = get_size(tr, tl)
    max_width = max(int(width_a), int(width_b))

    height_a = get_size(tr, br)
    height_b = get_size(tl, bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    rotation_matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, rotation_matrix, (max_width, max_height))
    return warped, rotation_matrix


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def apply_mask(card):
    im = card.image.copy()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA).astype(np.uint8)
    img = Image.new('L', im.shape[:2][::-1], 0)
    ImageDraw.Draw(img).polygon([tuple(x) for x in card.card_polygon], outline=255, fill=255)
    resized_mask = cv2.resize(np.array(img), np.array(im.shape[:2][::-1]) - 26).astype(np.uint8)
    resized_mask = np.pad(resized_mask, 13, 'constant', constant_values=0)
    im[:, :, 3] = np.array(resized_mask)
    return im


def extract_cards(files):
    for file in files:
        card = Card(file.split("/")[-1])
        annotations = json.load(open(file + ".json", 'r'))['shapes']
        for shape in annotations:
            if shape['label'] == 'card':
                card.set_card_polygon(np.array(shape['points']).astype(np.int32))
            else:
                card.add_label_polygon(shape['points'])
        image = cv2.imread(file + ".jpeg")
        card.set_image(image)
        img = apply_mask(card)
        extended_lines, intersection_points = get_bbox(card)
        img, rotation_matrix = four_point_transform(img, order_points(intersection_points))
        rotated_card_poly = rotate_polygon(card.card_polygon, rotation_matrix)
        rotated_label_poly = [rotate_polygon(polygon, rotation_matrix) for polygon in card.label_polygons]
        card.set_card_polygon(rotated_card_poly)
        card.set_label_polygons(rotated_label_poly)
        card.set_image(img)
        card.save()


if __name__ == '__main__':
    files = glob.glob("../data/raw_data/*.jpeg")
    files = [f.rstrip(".jpeg") for f in files]
    extract_cards(files)