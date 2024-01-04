import uuid
import time
import sys
import os

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, QualityForRecognition, APIErrorException

class FaceRecognizer:
    """
    A class for recognizing and identifying faces using Azure Cognitive Services.
    """

    def __init__(self, key, endpoint):
        """
        Initialize the FaceRecognizer with Azure credentials.

        Args:
            key (str): Azure vision key.
            endpoint (str): Azure vision endpoint.
        """
        self.face_client = FaceClient(endpoint, CognitiveServicesCredentials(key))
        self.person_group_id = str(uuid.uuid4())

    def create_person_group(self):
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

    def identify_faces(self, test_image_path):
        """
        Identify faces in a given image against the trained person group.

        Args:
            test_image_path (str): File path of the image in which faces are to be identified.

        Returns:
            None: Prints out identification results and verification.
        """
        print('Identifying faces in image:', test_image_path)
        time.sleep(10)  # Pausing to avoid rate limit on free account

        try:
            with open(test_image_path, 'rb') as image_stream:
                try:
                    # Detect Faces in the image
                    faces = self.face_client.face.detect_with_stream(
                        image=image_stream,
                        detection_model='detection_03',
                        recognition_model='recognition_04',
                        return_face_attributes=['qualityForRecognition']
                    )
                    if not faces:
                        print("No faces detected in the image.")
                        return

                    # Checks detected faces for suitable quality
                    face_ids = [face.face_id for face in faces if face.face_attributes.quality_for_recognition in [QualityForRecognition.high, QualityForRecognition.medium]]
                    if not face_ids:
                        print("No faces of suitable quality for recognition.")
                        return

                    # Attempts to identify the faces
                    results = self.face_client.face.identify(face_ids, self.person_group_id)
                    if not results:
                        print('No person identified in the person group.')
                        return

                    # Outputs all validated detections
                    for identified_face in results:
                        if len(identified_face.candidates) > 0:
                            print('Person is identified for face ID {} in image, with a confidence of {}.'.format(identified_face.face_id, identified_face.candidates[0].confidence))
                            verify_result = self.face_client.face.verify_face_to_person(
                                identified_face.face_id,
                                identified_face.candidates[0].person_id,
                                self.person_group_id
                            )
                            print('Verification result: {}. Confidence: {}'.format(verify_result.is_identical, verify_result.confidence))
                        else:
                            print('No person identified for face ID {} in image.'.format(identified_face.face_id))
                except APIErrorException as api_err:
                    print("API Error:", api_err.message)
                except Exception as e:
                    print("Error during API call:", str(e))
        except FileNotFoundError:
            print(f"The file {test_image_path} was not found.")
        except Exception as e:
            print("Error opening the image file:", str(e))


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
                    person = self.create_person(person_name)
                    print(f"Created person: {person_name}")

                    # Get all image file paths
                    image_urls = [os.path.join(person_path, img) for img in os.listdir(person_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

                    # Add faces to this person from each image
                    self.add_faces_to_person(person, image_urls)
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
            with open(image_path, 'rb') as image_stream:
                try:
                    detected_faces = self.face_client.face.detect_with_stream(
                        image=image_stream,
                        detection_model='detection_03',
                        recognition_model='recognition_04',
                        return_face_attributes=['qualityForRecognition']
                    )
                    for face in detected_faces:
                        if face.face_attributes.quality_for_recognition == QualityForRecognition.high:
                            self.face_client.person_group_person.add_face_from_stream(
                                self.person_group_id, person.person_id, image_stream
                            )
                            print(f"Face added to person {person.person_id} from image {image_path}")
                except APIErrorException as e:
                    print(f"Failed to process image {image_path}: {e.message}")
