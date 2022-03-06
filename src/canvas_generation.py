import copy
import glob
import json
import math
import multiprocessing
import random
from pathlib import Path

from shapely.geometry import Polygon

from transformations.BrightnessContrastTransform import BrightnessContrastTransform
from transformations.Compose import Compose
from transformations.PerspectiveTransform import PerspectiveTransform
from transformations.Resize import Resize
from utils import *


def rotate(image, angle_in_degrees, point=None):
    h, w = image.shape[:2]
    if not point:
        point = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(point, angle_in_degrees, 1)

    rad = math.radians(angle_in_degrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - point[0])
    rot[1, 2] += ((b_h / 2) - point[1])

    transformed_image = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return transformed_image


def overlay(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop


def check_polygons(card, metadata):
    local_meta = copy.deepcopy(metadata)
    p1 = Polygon(np.array(card['card_polygon']))
    for i in local_meta:
        corners = [Polygon(np.array(corner)) for corner in local_meta[i]['label_polygons']]
        corners = [p1.intersects(p2) or p1.crosses(p2) or p1.contains(p2) for p2 in corners]
        gone = 0
        for count, poly in enumerate(corners):
            if poly:
                if len(local_meta[i]['label_polygons']) == 1:
                    return False, metadata
                else:
                    del local_meta[i]['label_polygons'][count - gone]
                    gone += 1
    return True, local_meta


def random_card_generator(cards_paths, n):
    cards = cards_paths.copy()
    for card_path in cards[:n]:
        card = cv2.imread(card_path + ".png", -1)
        annots = json.load(open(card_path + ".json", 'r'))
        card_polygon = annots['card_polygon']
        label_polygons = annots['label_polygons']
        yield {'image': card, 'label_polygons': label_polygons, 'card_polygon': card_polygon,
               'filename': card_path.split("/")[-1]}


def translate_polygon(polygon, translation):
    return numpy_to_list(np.asarray(polygon) + translation)


def generate_random_point(max_y, max_x, card_y, card_x):
    x = random.randint(0, max_x - int(card_x))
    y = random.randint(0, max_y - int(card_y))
    return np.array((x, y))


def create_canvas(background, cards_paths, n=10):
    coords = []
    metadata = {}
    canvas = background[:, :, :3].copy()

    background_shape = canvas.shape[:2]

    for card_dict in random_card_generator(cards_paths, n):
        size = np.random.randint(int(0.4 * max(background_shape)), int(0.49 * max(background_shape)))
        augmented_card = transforms(card_dict, size=size)
        card = augmented_card['image']
        alpha_mask = card[:, :, 3] / 255.0
        card_rgb = card[:, :, :3]

        card_shape = card.shape[:2]
        tries = 0
        while True:
            repeat = False
            random_point = generate_random_point(*background_shape, *card_shape)
            new_card_poly = translate_polygon(augmented_card['card_polygon'], random_point)
            new_label_poly = [translate_polygon(polygon, random_point) for polygon in augmented_card['label_polygons']]
            card_polygons = {'label_polygons': new_label_poly, 'card_polygon': new_card_poly}
            polygons_ok, new_meta = check_polygons(card_polygons, metadata)
            if not polygons_ok:
                repeat = True
                tries += 1
            if not repeat:
                coords.append(random_point)
                overlay(canvas, card_rgb, *random_point, alpha_mask)
                metadata = new_meta
                metadata[card_dict['filename']] = card_polygons
                break
            if tries > 10:
                break

    return canvas, metadata


def polygon_to_yolo_bbox(polygon, image_shape):
    polygon = np.asarray(polygon)
    x_s, y_s = polygon[:, 0], polygon[:, 1]
    minX, maxX = x_s[x_s.argsort()][[0, -1]]
    minY, maxY = y_s[y_s.argsort()][[0, -1]]
    x_center = ((minX + maxX) / 2) / image_shape[1]
    y_center = ((minY + maxY) / 2) / image_shape[0]
    width = abs(minX - maxX) / image_shape[1]
    height = abs(minY - maxY) / image_shape[0]
    return np.float32([x_center, y_center, width, height])


def create_data(image, image_dict, filename):
    with open(filename.replace("images", "labels") + ".txt", 'w+') as fp:
        for card in image_dict:
            for polygon in image_dict[card]['label_polygons']:
                bbox = polygon_to_yolo_bbox(polygon, image.shape)
                fp.write(str(class_to_id[card.upper()]) + " " + " ".join(list(map(str, bbox))) + '\n')
    cv2.imwrite(filename + ".jpg", image)


def _generate_dataset(back_path, cards, root_dir, i):
    random.shuffle(cards)
    back = cv2.imread(back_path, -1)
    back = cv2.resize(back, (1920, 1920))
    n = random.randint(1, 8)
    #         cards = cards[n:] + cards[:n]
    out, image_dict = create_canvas(back, cards, n)
    resized_data = resize({'image': out, 'image_dict': image_dict}, 608, True)
    aug_image, aug_image_dict = resized_data['image'], resized_data['image_dict']
    create_data(aug_image, aug_image_dict, root_dir + f"image_{i}")


def generate_dataset(root_dir, cards, textures, size):
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    Path(root_dir.replace("images", "labels")).mkdir(parents=True, exist_ok=True)
    random.shuffle(cards)
    back_paths = np.random.choice(textures, size=size)
    cards = [cards] * size
    root_dirs = [root_dir] * size
    ids = list(range(size))
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    pool.starmap(_generate_dataset, list(zip(back_paths, cards, root_dirs, ids)))


if __name__ == '__main__':
    dataset_size = 20000
    dataset_path = "../datasets/cards/images/train20000_squares/"
    resize = Resize()
    transforms = Compose([
        PerspectiveTransform(degrees_range=(0, 110)),
        BrightnessContrastTransform(),
        resize
    ])

    textures = glob.glob("../data/dtd/images/*/*.jpg")
    cards = glob.glob("../data/interim_data/*.png")
    cards = [f.rstrip(".png") for f in cards]
    with open("class_mapping.json", 'r') as fp:
        class_to_id = json.loads(fp.read())

    generate_dataset(dataset_path, cards, textures, dataset_size)