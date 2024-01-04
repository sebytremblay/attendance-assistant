import os

from FaceRecognizer import FaceRecognizer
from FaceDetection import FaceDetection

class Controller:
    """
    A class for recognizing and identifying faces using GCP and Azure Cognitive Services.
    """

    def __init__(self, api_key, api_endpoint):
        """
        Initializes a controller to parse faces from a video and identify them.

        Args:
            api_key (str): Azure vision key.
            api_endpoint (str): Azure vision endpoint.
        """
        # Initializes recognition model
        self.face_recognizer = FaceRecognizer(api_key, api_endpoint)
        self.face_recognizer.create_person_group()

    def train_model(self, training_data_path):
        """
        Traings the model on the data in the specified path.

        Args:
            training_data_path (str): the file path to the directory containing the training data.
        """
        self.face_recognizer.add_training_data(training_data_path)
        self.face_recognizer.train_person_group()

    def analyze_video(self, video_path, fps):
        """
        Analyzes a given video to detect and identify faces. It first runs face detection
        on each frame of the video at the specified frames per second (fps). It then
        iterates through each detected face and attempts to identify it using the
        face recognizer.

        Args:
            video_path (str): The file path of the video to analyze.
            fps (int): The number of frames per second to process in the video.
        """
        # Creates new detection object and detects faces
        face_detector = FaceDetection(video_path, fps)
        face_detector.run()
        
        # Retrieve all image file paths from the output directory
        for image_file in os.listdir(face_detector._FaceDetection__output_folder):
            # Identifies faces on each image
            image_path = os.path.join(face_detector._FaceDetection__output_folder, image_file)
            self.face_recognizer.identify_faces(image_path)       