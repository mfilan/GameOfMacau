import glob
import cv2
from PIL import Image
import re
import copy

class Params:
    def __init__(self):
        self.current = False
        self.war = False
        self.detected = set()
        self.f_count = 0
        self.war_val = 0
        self.block_val = 0
        self.play = ''
        self.msg = ''

def draw_box(frame, box):
    box = [float(i) for i in box]
    h, w, _ = frame.shape
    mid_x, mid_y = w * box[0], h * box[1]
    start_point = (int(mid_x - box[2] * w / 1.2), int(mid_y - box[3] * h / 1.2))
    end_point = (int(mid_x + box[2] * w / 1.2), int(mid_y + box[3] * h / 1.2))
    cv2.rectangle(frame, start_point, end_point, (0, 0, 255), thickness=4, lineType=cv2.LINE_8)

def get_frame_info(path):
    cards = {}
    with open(path, 'r') as infile:
        for line in infile:
            label = card_names[int(line.split()[0])]
            box = line.split()[1:]
            if label not in cards:
                cards[label] = {}
                cards[label]['count'] = 1
                cards[label]['boxes'] = [box]
            else:
                cards[label]['count'] += 1
                cards[label]['boxes'].append(box)
    return copy.deepcopy(cards), label

def handle_events(label, params):
    value, suite = label[:-1], label[-1]

    #Jack & Queen
    if value == 'Q':
        params.msg = f'Queen on everything, everything on Queen.'
    if value == 'J':
        params.msg = f'Player {params.current} demands something.'

    #Handling war
    if value == '2' or value =='3' or label == 'KH' or label == 'KS':
        if not params.war:
            params.war = True
            if label == 'KH' or label == 'KS':
                params.war_val = 5
            else:
                params.war_val = int(value)
            params.msg = f"War has started! value: {params.war_val}"

        else:
            if label == 'KH' or label == 'KS':
                params.war_val += 5
            else:
                params.war_val += int(value)
            params.msg = f"War continues! value: {params.war_val}"

    elif params.war:
        params.war = False
        params.msg = f'War has ended, final value: {params.war_val}'

    #Handling block
    if value =='4':
        params.block_val += 1
        params.msg = f'Player {params.current} will be blocked for {params.block_val} turns!'

    elif params.block_val:
        params.block_val = 0

def get_frame_number(path):
    path = path.split("/")[-1]
    path = path.rstrip(".txt")
    path = path.split("_")[-1]
    return int(path)

def annotate(frame, params):
    cv2.putText(frame, params.play,
                    (0, 50),
                    font, 1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_4)
    cv2.putText(frame, params.msg,
                    (0, 100),
                    font, 1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_4)

if __name__ == "__main__":
    numbers = re.compile(r'(\d+)')

    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    games = glob.glob("./videos/*.mp4")
    print(games)
    print()
    labeled = glob.glob("./yolov5/runs/detect/exp*/labels/*_1.txt")
    labeled = [f.rstrip("_1.txt") for f in labeled]
    print(labeled)

    card_names = ['10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S',
                  '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S', '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C',
                  '9D', '9H', '9S', 'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS', 'QC', 'QD',
                  'QH', 'QS']

    font = cv2.FONT_HERSHEY_SIMPLEX

    for file_path in labeled:
        labels = glob.glob(file_path + '*.txt')
        vid_path = file_path.split("/")[-1] + '.mp4'
        in_path = './videos/' + vid_path
        out_path = './annotated/' + vid_path

        params = Params()

        print(in_path)
        cap = cv2.VideoCapture(in_path)
        result = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), 20, (608, 608))
        con = 0
        for path in sorted(labels, key=numericalSort):
            con += 1
            ret, frame = cap.read()
            frame_num = get_frame_number(path)
            while (con < frame_num):
                con += 1
                annotate(frame, params)
                result.write(frame)
                ret, frame = cap.read()
            cards, label = get_frame_info(path)

            for box in cards[label]['boxes']:
                draw_box(frame, box)

            for label in cards:
                if cards[label]['count'] >= 3:
                    params.f_count += 1
                    if params.f_count >= 3:
                        value, suite = label[:-1], label[-1]
                        if label not in params.detected:
                            params.detected.add(label)
                            params.play = f'Player {params.current} plays {label}.'
                            params.msg = ''
                            params.current = not params.current
                            handle_events(label, params)
                else:
                    params.f_count = 0

            annotate(frame, params)
            result.write(frame)
            # print(params.detected)
        cap.release()
        result.release()
        # print(sorted(labels, key=numericalSort)[-1], con)