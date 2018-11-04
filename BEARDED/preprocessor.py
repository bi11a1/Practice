import cv2
import os
from tqdm import tqdm

ALL_EMOTIONS = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]
JAFFE_EMOTIONS = ["NE", "AN", "DI", "FE", "HA", "SA", "SU"]

RAW_TRAIN_DIR = 'raw_train' # Directory for raw train images
RAW_TEST_DIR = 'raw_test' # Directory for raw test images
TRAIN_DIR = 'train' # Directory for processed train images
TEST_DIR = 'test' # Directory for processed test images

# --------------------------------------------------------------------------------------------------------
# Given a rgb image crop face portion and then performs histogram equalization
# Returns Grayscale face image
# Should contain image of exactly one person
# Need to apply cropping of image in recognizer.py according to choosen IMG_SIZE
def preprocess_img(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faceDet_one = cv2.CascadeClassifier("face_classifiers/haarcascade_frontalface_default.xml")
    faceDet_two = cv2.CascadeClassifier("face_classifiers/haarcascade_frontalface_alt2.xml")
    faceDet_three = cv2.CascadeClassifier("face_classifiers/haarcascade_frontalface_alt.xml")
    faceDet_four = cv2.CascadeClassifier("face_classifiers/haarcascade_frontalface_alt_tree.xml")

    face_one = faceDet_one.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(face_one) == 1:
        facefeatures = face_one
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        output_img = cv2.equalizeHist(gray_img)
        return output_img

    for (x, y, w, h) in facefeatures:
        gray_img = gray_img[y:y+h, x:x+w]
        output_img = cv2.equalizeHist(gray_img)
        return output_img

# --------------------------------------------------------------------------------------------------------
# Given the labeled image in RAW_TRAIN_DIR it creates seven emotion folders in TRAIN_DIR
# For jaffe databse
def create_train_images():
    # Creating all emotions folder in the training directory
    for emotion in ALL_EMOTIONS:
        emotion_dir = os.path.join(TRAIN_DIR, emotion)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

    # Processing image and saving in train directory
    for img in tqdm(os.listdir(RAW_TRAIN_DIR)):
        emotion = img[-7]+img[-6]
        img_path = os.path.join(RAW_TRAIN_DIR, img)
        img_data = cv2.imread(img_path)
        img_data = preprocess_img(img_data)
        for idx, val in enumerate(JAFFE_EMOTIONS):
            if(val == emotion):
                save_img_path = os.path.join(TRAIN_DIR, ALL_EMOTIONS[idx])
                save_img_path = os.path.join(save_img_path, img)
                cv2.imwrite(save_img_path, img_data)

# --------------------------------------------------------------------------------------------------------
# Given the unlabeled image in RAW_TEST_DIR it saves processed images in TEST_DIR
# Need to apply cropping of image in recognizer.py according to choosen IMG_SIZE
def create_test_images():
    for img in tqdm(os.listdir(RAW_TEST_DIR)):
        img_path = os.path.join(RAW_TEST_DIR, img)
        img_data = cv2.imread(img_path)
        img_data = preprocess_img(img_data)
        save_img_path = os.path.join(TEST_DIR, img)
        cv2.imwrite(save_img_path, img_data)

# --------------------------------------------------------------------------------------------------------
# create_train_images()
create_test_images()