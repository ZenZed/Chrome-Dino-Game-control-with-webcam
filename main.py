import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2
import pyautogui


CMD = ["Jump", "Duck", "None"]


def tm():
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rev, frame = vc.read()
        # img_name = "opencv_frame_0.jpg"
        # cv2.imwrite(img_name, frame)


    while rev:
        rev, frame = vc.read()
        cv2.imshow("MOK", frame)
        if cv2.waitKey(1) == 25:
            break

        # image = Image.open(img_name)
        #
        # # resize the image to a 224x224 with the same strategy as in TM2:
        # # resizing the image to be at least 224x224 and then cropping from the center
        # size = (224, 224)
        # image = ImageOps.fit(image, size, Image.ANTIALIAS)

        # turn the image into a numpy array
        image_array = cv2.resize(frame, (224,224),  interpolation = cv2.INTER_AREA)

        # display the resized image
        # image.show()

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        # classify the image into cmds
        command = CMD[prediction[0].argmax(axis=0)]
        # debugging puposes
        print(command)
        # Practical output
        if command == "Jump":
            pyautogui.press('space')
        elif command == "Duck":
            pyautogui.press('down')




tm()

