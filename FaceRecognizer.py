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

class FaceRecognizer:
    def __init__(self, key, endpoint, person_group_id=""):
        """
        Initialize the FaceRecognizer with Azure credentials.

        Args:
            key (str): Azure vision key.
            endpoint (str): Azure vision endpoint.
        """
        self.face_client = FaceClient(endpoint, CognitiveServicesCredentials(key))
        
        if not person_group_id:
            self.person_group_id = str(uuid.uuid4())
            self.__initialize_person_group()
        else:
            self.person_group_id = person_group_id
        
    def __initialize_person_group(self):
        """
        Create a person group for face recognition.
        """
        print('Creating Person Group:', self.person_group_id)
        self.face_client.person_group.create(
            person_group_id=self.person_group_id,
            name=self.person_group_id,
            recognition_model='recognition_04'
        )
        
    def delete_person_group(self):
        """
        Delete a person group for face recognition.
        """
        self.face_client.person_group.delete(
            person_group_id=self.person_group_id
        )
        print("Deleted the person group {} from the source location.".format(self.person_group_id))

    def create_person(self, name):
        """
        Create a person in the person group.

        Args:
            name (str): Name of the person to be added.

        Returns:
            Person: The created person object.
        """
        return self.face_client.person_group_person.create(self.person_group_id, name)
    
    def train_person_group(self):
        """
        Train the person group with added persons and faces.
        """
        print("Training the person group:", self.person_group_id)
        self.face_client.person_group.train(self.person_group_id)

        while True:
            training_status = self.face_client.person_group.get_training_status(self.person_group_id)
            print("Training status: {}.".format(training_status.status))
            if training_status.status is TrainingStatusType.succeeded:
                break
            elif training_status.status is TrainingStatusType.failed:
                self.face_client.person_group.delete(person_group_id=self.person_group_id)
                sys.exit('Training the person group has failed.')
            time.sleep(5)
            
    def detect_faces(self, frame_buffer):
        """Detects faces in an image.

        Args:
            frame_buffer (BufferedReader): the buffer containing the image data.
            
        Returns:
            DetectedFace[]: the faces detected in the image.
        """
        print('Detecting faces in image buffer')
        time.sleep(10)  # Pausing to avoid rate limit on free account

        try:
            # Detect Faces in the image
            faces = self.face_client.face.detect_with_stream(
                image=frame_buffer,
                detection_model='detection_03',
                recognition_model='recognition_04',
                return_face_attributes=['qualityForRecognition']
            )
            
            print(f"Detected {len(faces)} faces in image buffer.\n")
            return faces
        except APIErrorException as api_err:
            print(f"API Error:{api_err.message}\n")
        except Exception as e:
            print(f"Error during API call: {e}\n")

    def identify_faces(self, frame_buffer):
        """
        Identify faces in a given frame buffer against the trained person group.

        Args:
            frame_buffer (BufferedReader): Data for the image to process.

        Returns:
            IdentifyResult[]: all identified persons in image. 
        """
        time.sleep(10)  # Pausing to avoid rate limit on free account

        # Detects faces in the buffer
        faces = self.detect_faces(frame_buffer)

        # Checks detected faces for suitable quality
        accepted_quality = [QualityForRecognition.high, QualityForRecognition.medium]
        face_ids = [
            face.face_id for face in faces 
            if face.face_attributes.quality_for_recognition in accepted_quality]
        if not face_ids:
            print("No faces of suitable quality for recognition.\n")
            return []

        # Attempts to identify the faces
        results = self.face_client.face.identify(face_ids, self.person_group_id)
        print(f"Identified {len(results)} person(s) from the person group.\n")
        return results
    
    def add_training_data(self, training_data_directory):
        """
        Add training data from the given directory.

        Args:
            training_data_directory (str): Path to the training data directory.
        """
        if not os.path.exists(training_data_directory):
            raise ValueError("Training Directory Does Not Exist")
        
        for person_name in os.listdir(training_data_directory):
            person_path = os.path.join(training_data_directory, person_name)
            if os.path.isdir(person_path):
                # Create a person for the subfolder
                try:
                    # Get all image file paths
                    image_urls = [os.path.join(person_path, img) for img in os.listdir(person_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    
                    # If training data for this person exists, create a new person
                    if len(image_urls) > 0:
                        person = self.create_person(person_name)
                        self.add_faces_to_person(person, image_urls)
                        
                        print(f"Created person: {person_name}")
                except APIErrorException as e:
                    print(f"Failed to add {person_name}: {e.message}")

    def add_faces_to_person(self, person, image_paths):
        """
        Add faces to a person from given file paths.

        Args:
            person (Person): The person object to which the faces will be added.
            image_paths (list of str): List of image file paths containing the person's face.
        """
        for image_path in image_paths:
            # Prevent call rate limit excess
            time.sleep(5)
            
            try:
                with open(image_path, 'rb') as image_stream:
                    detected_faces = self.face_client.face.detect_with_stream(
                        image=image_stream,
                        detection_model='detection_03',
                        recognition_model='recognition_04',
                        return_face_attributes=['qualityForRecognition']
                    )
                    for face in detected_faces:
                        if face.face_attributes.quality_for_recognition == QualityForRecognition.high \
                            or face.face_attributes.quality_for_recognition == QualityForRecognition.medium:
                            # Reset the stream position to the beginning after reading it for size
                            image_stream.seek(0)
                            
                            self.face_client.person_group_person.add_face_from_stream(
                                self.person_group_id, person.person_id, image_stream
                            )
                            print(f"Face added to person {person.person_id} from image {image_path}")
            except APIErrorException as e:
                print(f"Failed to process image {image_path}: {e.message}")
            except Exception as e:
                # Adding a generic exception catch for other potential errors
                print(f"An error occurred with image {image_path}: {e}")