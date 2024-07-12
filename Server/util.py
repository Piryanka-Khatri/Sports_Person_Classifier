import joblib
import numpy as np
import base64
from wavelet import w2d
import json
import cv2
import sklearn

__class_name_to_number = {}
__class_number_to_name = {}

__model = None


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]


def classify_image(image_base64_data, file_path=None):
    result = []
    images = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    for image in images:
        resized_img = cv2.resize(image, (32, 32))
        img_har = w2d(image, 'db1', 5)
        resized_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((resized_img.reshape(32 * 32 * 3, 1), resized_img_har.reshape(32 * 32, 1)))

        len_image_array = 32 * 32 * 3 + 32 * 32

        final = combined_img.reshape(1, len_image_array).astype(float)

        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final)*100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result


def get_cv2_image_from_base64_string(base64_str):
    encoded_data = base64_str.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    cropped_faces = []

    face_cascade = cv2.CascadeClassifier("../opencv/data/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("../opencv/data/haarcascade_eye.xml")

    if image_path:
        image = cv2.imread(image_path)
    else:
        image = get_cv2_image_from_base64_string(image_base64_data)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray_image)

    for (x, y, w, h) in face:
        roi_gray_image = gray_image[y:y+h, x:x+h]
        roi_image = image[y:y + h, x:x + h]
        eye = eye_cascade.detectMultiScale(roi_gray_image)
        if len(eye) >= 2:
            cropped_faces.append(roi_image)
    return cropped_faces


def get_msd_image():
    with open("test.txt") as f:
        return f.read()


def load_saved_artifacts():
    global __model
    global __class_name_to_number
    global __class_number_to_name

    print("loading saved artifacts...start")

    with open("./artifacts/column_directory.json" , 'r') as file:
        __class_name_to_number = json.load(file)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    if __model is None:
        with open("./artifacts/saving_the_model.pkl", "rb") as f:
            __model = joblib.load(f)

    print("loading saved artifacts...done")


if __name__ == "__main__":
    load_saved_artifacts()
    # print(classify_image(get_msd_image(), None))
    print(classify_image(None, "test_images/09Hunter1-superJumbo.jpg"))
    print(classify_image(None,"test_images/184-1847379_download-maria-sharapova-wallpaper-2013-wallpaper-hd-full.jpg"))
    print(classify_image(None, "test_images/20190804-The18-Image-Lionel-Messi-Speech-Camp-Nou-2019-1280x720.jpg"))
    print(classify_image(None, "test_images/85420dacb0.jpg"))
    print(classify_image(None, "test_images/FEDERER_16b9e25e144_large.jpg"))
    print(classify_image(None, "test_images/images (4).jpeg"))
    print(classify_image(None, "test_images/2.jpeg"))


