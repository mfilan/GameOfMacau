import glob

import cv2


def resize_image(image, width, height, colour=(0, 0, 0)):
    h, w, layers = image.shape
    if h > height:
        ratio = height / h
        image = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
    h, w, layers = image.shape
    if w > width:
        ratio = width / w
        image = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
    h, w, layers = image.shape
    if h < height and w < width:
        hless = height / h
        wless = width / w
        if hless < wless:
            image = cv2.resize(image, (int(image.shape[1] * hless), int(image.shape[0] * hless)))
        else:
            image = cv2.resize(image, (int(image.shape[1] * wless), int(image.shape[0] * wless)))
    h, w, layers = image.shape
    if h < height:
        df = height - h
        df /= 2
        image = cv2.copyMakeBorder(image, int(df), int(df), 0, 0, cv2.BORDER_CONSTANT, value=colour)
    if w < width:
        df = width - w
        df /= 2
        image = cv2.copyMakeBorder(image, 0, 0, int(df), int(df), cv2.BORDER_CONSTANT, value=colour)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def resize_video(in_path, out_path, size=(608, 608)):
    vidcap = cv2.VideoCapture(in_path)
    result = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (608, 608))
    while True:
        success, image = vidcap.read()
        if success:
            resized = resize_image(image, *size)
            result.write(resized)
        else:
            break
    vidcap.release()
    result.release()


if __name__ == "__main__":
    paths = glob.glob("../data/interim_videos/*.mp4")
    for vid_path in paths:
        print(vid_path)
        resize_video(vid_path, './out/' + vid_path.split("/")[-1])