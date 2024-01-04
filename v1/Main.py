import os
import credentials

from Controller import Controller

if __name__ == "__main__":
    TRAINING_DATA_PATH = "./training_data"
    
    controller = Controller(credentials.API_KEY, credentials.API_ENDPOINT)
    controller.train_model(TRAINING_DATA_PATH)
    
    video_path = "./examples/test3.MOV"
    fps = 29.99
    controller.analyze_video(video_path, fps)