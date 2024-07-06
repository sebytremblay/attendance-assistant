# from ..credentials import API_KEY, API_ENDPOINT

from Controller import Controller
from FaceDetection import FaceDetection

if __name__ == "__main__":
    # TRAINING_DATA_PATH = "./training_data"
    
    # controller = Controller(credentials.API_KEY, credentials.API_ENDPOINT)
    # controller.train_model(TRAINING_DATA_PATH)
    
    # video_path = "./examples/test3.MOV"
    # fps = 29.99
    # controller.analyze_video(video_path, fps)
    
    video_path = "./examples/test5.MOV"
    fps = 29.99
    
    face_detector = FaceDetection(video_path, fps, face_frame_padding=50)
    face_detector.run_face_detection()