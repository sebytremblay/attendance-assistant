import uuid
import time
import sys
import os
import cv2

from io import BytesIO
from PIL import Image
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, QualityForRecognition, APIErrorException

class FaceAnnotation:
    def __init__(self, key, endpoint, person_group_id=""):
        """
        Initialize the FaceRecognizer with Azure credentials.

        Args:
            key (str): Azure vision key.
            endpoint (str): Azure vision endpoint.
        """
        self.face_client = FaceClient(endpoint, CognitiveServicesCredentials(key))