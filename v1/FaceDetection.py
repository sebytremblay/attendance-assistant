import io
import cv2
import os
import shutil

from google.cloud import videointelligence_v1 as videointelligence

class FaceDetection:
    def __init__(self, video_path, fps, output_folder = "./output", purge_output_on_launch = True, face_frame_padding = 25):
        """Capture a specific frame from a video and save the detected face.

        Args:
            video_path (str): The file path of the video to analyze.
            fps (float): The FPS for the video.
            output_folder (str): The file path for output destination.
            purge_output_on_launch (bool): Whether to clear the output destination before saving.
            face_frame_padding (float): The amount of padding to add to the face frame border.
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.__video_path = video_path
        self.__fps = fps
        self.__output_folder = f"{output_folder}/{video_name}"
        self.__purge_output_on_launch = purge_output_on_launch
        self.__face_frame_padding = face_frame_padding
        
        self.__clear_output_folder()

    def __clear_output_folder(self):
        """Clear all files in the output folder."""
        if not self.__purge_output_on_launch:
            return
        
        if os.path.exists(self.__output_folder):
            shutil.rmtree(self.__output_folder)
        os.makedirs(self.__output_folder, exist_ok=True)

    def __get_time(self, time_offset):
        """Calculate the time in seconds from a time_offset object.

        Args:
            time_offset (TimeOffset): The time offset object from video annotation.

        Returns:
            float: Time in seconds.
        """
        return time_offset.seconds + time_offset.microseconds / 1e6

    def __capture_and_save_frame(self, cap, frame_num, box, descriptor):
        """Capture a specific frame from a video and save the detected face.

        Args:
            cap (cv2.VideoCapture): Video capture object.
            frame_num (int): Frame number to capture.
            box (NormalizedBoundingBox): Bounding box for the face in the frame.
            descriptor (str): Descriptor for the frame (e.g., 'first', 'middle', 'last').
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture {descriptor} frame at timestamp:", frame_num / self.__fps)
            return

        face_frame = self.__crop_face_frame(frame, box)
        face_frame_filename = f"{self.__output_folder}/face_frame_{descriptor}_at_{frame_num}_box_{box.left}_{box.top}_{box.right}_{box.bottom}.jpg"
        cv2.imwrite(face_frame_filename, face_frame)
        print(f"Saved {face_frame_filename}")

    def __crop_face_frame(self, frame, box):
        """Crop the face frame from the given frame based on the bounding box.

        Args:
            frame (numpy.ndarray): The frame from which to crop the face.
            box (NormalizedBoundingBox): The bounding box of the face in the frame.

        Returns:
            numpy.ndarray: The cropped face frame.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(box.left * w), int(box.top * h), int(box.right * w), int(box.bottom * h)

        x1_border = max(x1 - self.__face_frame_padding, 0)
        y1_border = max(y1 - self.__face_frame_padding, 0)
        x2_border = min(x2 + self.__face_frame_padding, w - 1)
        y2_border = min(y2 + self.__face_frame_padding, h - 1)

        return frame[y1_border:y2_border, x1_border:x2_border]

    def __detect_faces(self):
        """Detects faces in a video from a local file.

        Returns:
            AnnotatedVideoResponse: The response containing the face detection annotations.
        """
        client = videointelligence.VideoIntelligenceServiceClient()

        with io.open(self.__video_path, "rb") as f:
            input_content = f.read()

        # Configure the request
        config = videointelligence.FaceDetectionConfig(
            model="builtin/latest",
            include_bounding_boxes=True, 
            include_attributes=True
        )
        context = videointelligence.VideoContext(face_detection_config=config)

        # Start the asynchronous request
        operation = client.annotate_video(
            request={
                "features": [videointelligence.Feature.FACE_DETECTION],
                "input_content": input_content,
                "video_context": context,
            }
        )

        print("\nProcessing video for face detection annotations.")
        result = operation.result(timeout=300)
        print("\nFinished processing.\n")

        # Return the first result, because a single video was processed.
        return result.annotation_results[0]
    
    def save_detected_faces(self):
        """Detect and save the first, middle, and last frames of faces in a video."""
        
        annotation_result = self.__detect_faces()
        cap = cv2.VideoCapture(self.__video_path)
        if not cap.isOpened():
            print("Error opening video file.")
            return

        for annotation in annotation_result.face_detection_annotations:
            for track in annotation.tracks:
                start_offset_seconds = self.__get_time(track.segment.start_time_offset)
                end_offset_seconds = self.__get_time(track.segment.end_time_offset)
                midpoint_offset_seconds = (start_offset_seconds + end_offset_seconds) / 2

                for timestamp, descriptor in [
                    (start_offset_seconds, 'first'), 
                    (midpoint_offset_seconds, 'middle'), 
                    (end_offset_seconds, 'last')
                ]:
                    # Find the timestamped object closest to the descriptor
                    closest_timestamped_object = min(track.timestamped_objects,
                                                    key=lambda obj: abs(
                                                        self.__get_time(obj.time_offset) - timestamp))
                    
                    # Calculate frame number based on the midpoint time offset of the object
                    frame_num = int(self.__get_time(closest_timestamped_object.time_offset) * self.__fps)

                    # Use the bounding box from the closest timestamped object
                    box = closest_timestamped_object.normalized_bounding_box
                    
                    # Saves the cropped image
                    self.__capture_and_save_frame(cap, frame_num, box, descriptor)

        cap.release()