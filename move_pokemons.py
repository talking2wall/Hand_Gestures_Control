import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


can_drag = False
focused_fig_id = -1

# place overlay image
def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

def Pinch(x1, y1, x2, y2):

    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)

    if (math.sqrt(pow(x2-x1,2) + pow(y2-y1,2)) <= 23):
        return True

    return False

def InsideBox(x, y, x_start, y_start, fig_height, fig_width):
    # x,y ---> hand position
    # x_start, y_start ---> box start position
    # fig_height ---> image height
    # fig_width ---> image width

    if (x > x_start and x < (x_start + fig_width) and
        y > y_start and y < (y_start + fig_height)):
        return True
    
    return False

class Figure:

    gId = 0

    def __init__(self, x, y, image_path, height, width):
        self.x = x
        self.y = y
        self.image = cv2.imread(str(image_path), -1)
        self.height = int(height)
        self.width = int(width)
        self.id = Figure.gId
        Figure.gId+=1


cap = cv2.VideoCapture(0)
cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(cap_width, cap_height)


Figures = []
Figures.append(Figure(250,10,"images/pik.png",150,150)) ## Pickachu Init
Figures.append(Figure(10,10,"images/bul.png",150,150)) ## Bulbasaur Init
Figures.append(Figure(480,10,"images/cha.png",150,150)) ## Charmander Init


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

                INDEX_FINGER = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * cap_width,
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * cap_height)

                THUMB_FINGER = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * cap_width,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * cap_height)

                
                if (Pinch(THUMB_FINGER[0], THUMB_FINGER[1], INDEX_FINGER[0], INDEX_FINGER[1])):
                    can_drag = True
                else:
                    can_drag = False
                    focused_fig_id = -1


                if can_drag:
                    for fig in Figures:
                        if InsideBox(THUMB_FINGER[0], THUMB_FINGER[1], fig.x, fig.y, fig.height, fig.width):
                            
                            print(focused_fig_id)

                            if (focused_fig_id == -1):
                                focused_fig_id = fig.id # lock on this figure

                            if (fig.id != focused_fig_id):
                                continue
                            else:
                                fig.x = int(THUMB_FINGER[0]) - 75
                                fig.y = int(THUMB_FINGER[1]) - 75


        for fig in Figures:
            add_transparent_image(image, fig.image, fig.x, fig.y)

        cv2.imshow("hi", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
                
cap.release()