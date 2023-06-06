import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class ConvLSTM():


    def predict_ConvLSTM( video_file_path):

        IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
        SEQUENCE_LENGTH = 20
        CLASSES_LIST = ["BabyCrawling", "BaseballPitch", "HorseRace", "PlayingGuitar", "Skiing", "WalkingWithDog","Rafting","Surfing","CuttingInKitchen","Typing","Haircut","Fencing","LongJump","PushUps","Swing","YoYo","SkyDiving","ApplyLipstick","Billiards","FrisbeeCatch"] 


        ConvLSTM_model = load_model('convlstm_model_mod___Date_Time_2022_05_27__04_49_49___Loss_1.4444868564605713___Accuracy_0.6968504190444946.h5')
        video_reader = cv2.VideoCapture(video_file_path)
        frames_list = []
        predicted_class_name = ''
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
        for frame_counter in range(SEQUENCE_LENGTH):
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
            success, frame = video_reader.read()
            if not success:
                break
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255
            frames_list.append(normalized_frame)

        predicted_labels_probabilities = ConvLSTM_model.predict(
            np.expand_dims(frames_list, axis=0))[0]
        predicted_label = np.argmax(predicted_labels_probabilities)
        predicted_class_name = CLASSES_LIST[predicted_label]
        # ans = []

        # idx = (-predicted_labels_probabilities).argsort()[:5]

        # for predicted_label in idx:
        #     predicted_class_name = CLASSES_LIST[predicted_label]
        #     ans.append(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        video_reader.release()
        return predicted_class_name, float("{:.3f}".format(predicted_labels_probabilities[predicted_label] * 100))
