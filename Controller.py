from v2.FaceRecognizer import FaceRecognizer

class Controller:
    """
    A class for recognizing and identifying faces using GCP and Azure Cognitive Services.
    """

    def __init__(self, api_key, api_endpoint, person_group_id=""):
        """
        Initializes a controller to parse faces from a video and identify them.

        Args:
            api_key (str): Azure vision key.
            api_endpoint (str): Azure vision endpoint.
        """
        # Initializes recognition model
        self.face_recognizer = FaceRecognizer(api_key, api_endpoint, person_group_id)

    def train_model(self, training_data_path):
        """
        Traings the model on the data in the specified path.

        Args:
            training_data_path (str): the file path to the directory containing the training data.
        """
        self.face_recognizer.add_training_data(training_data_path)
        self.face_recognizer.train_person_group()

    def analyze_video(self, video_path, frame_interval, train_model=False, training_data_path=""):
        """
        Analyzes a given video to detect and identify faces. It first runs face detection
        on each frame of the video at the specified frames per second (fps). It then
        iterates through each detected face and attempts to identify it using the
        face recognizer.

        Args:
            video_path (str): The file path of the video to analyze.
            frame_interval (int): The rate to iterate through the video at
            train_model (bool): Whether to train the model before processing.
            training_data_path (str): The file path to the training data.
        """
        # Trains the model if requested
        if train_model:
            self.train_model(training_data_path)
            
        # Analyze the video and produce attendance list
        self.face_recognizer.recognize_faces(video_path, frame_interval)