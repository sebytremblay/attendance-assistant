import io
import cv2
import pprint

from FaceDetector import FaceDetector

class VideoAnalytics:
    def __init__(self, recognition_model, detection_model=None):
        """Analyzes and annotates a video with facial recognition.

        Args:
            recognition_model (FaceRecognizer): the recognition model to use for video analytics.
            detection_model (FaceDetector): the detection model to use for video analytics.
        """
        self.recognition_model = recognition_model
        if detection_model is None:
            self.detection_model = FaceDetector()
        else:
            self.detection_model = detection_model

    def __get_time(self, time_offset):
        """Calculate the time in seconds from a time_offset object.

        Args:
            time_offset (TimeOffset): The time offset object from video annotation.

        Returns:
            float: Time in seconds.
        """
        return time_offset.seconds + time_offset.microseconds / 1e6        

    def __crop_face_frame(self, cap, frame_num, box, padding=25):
        """Crop the face frame from the given frame based on the bounding box.

        Args:
            cap (cv2.VideoCapture): The video capture object.
            frame_num (int): The frame number to extract.
            box (NormalizedBoundingBox): The bounding box of the face in the frame.
            padding (int): The number of pixels to pad the bounding box by.

        Returns:
            io.BufferedReader: The cropped face frame as a buffer reader.
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(box.left * w), int(box.top * h), int(box.right * w), int(box.bottom * h)

        x1_border = max(x1 - padding, 0)
        y1_border = max(y1 - padding, 0)
        x2_border = min(x2 + padding, w - 1)
        y2_border = min(y2 + padding, h - 1)

        cropped_frame = frame[y1_border:y2_border, x1_border:x2_border]
        buffer = io.BytesIO()
        buffer.write(cv2.imencode('.jpg', cropped_frame)[1].tobytes())
        buffer.seek(0)

        return io.BufferedReader(buffer)

    def extract_video_settings(self, cap):
        """Extracts relevant information from an OpenCV video.
        
        Args:
            cap (cv2.VideoCapture): The video capture object.
            
        Returns:
            TODO"""
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{output_path}/annotated_video.mp4', fourcc, fps, (width, height))

        return width, height, fps, fourcc, out
        
    def annotate_video(self, video_path, output_path, annotation_interval, video_annotations=None):
        """Annotate the entire video, adding bounding boxes to faces when they appear on screen.
        
        Args:
            video_path (str): The path to the video file.
            output_path (str): The path to the output directory.
            annotation_interval (int): The number of frames between each face detection.
            video_annotations (AnnotatsedVideoResponse): The video annotations from the Video Intelligence API."""        
        # Open the original video and prepare a VideoWriter object to output the annotated video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file.")
            return

        # Store relevant video information
        width, height, fps, fourcc, out = extract_video_settings(cap)

        # Extract all face tracks
        if video_annotations is None:
            video_annotations = self.detection_model.detect_faces(video_path)

        # Prepare a mapping of frame numbers to face bounding boxes
        face_frames = {}
        pprint.pprint(video_annotations.face_detection_annotations)
        len_annotations = len(video_annotations.face_detection_annotations)
        for annotation in video_annotations.face_detection_annotations:
            pprint.pprint(annotation.tracks)
            len_tracks = len(annotation.tracks)
            for track in annotation.tracks:
                pprint.pprint(track.timestamped_objects)
                len_timestamped_objects = len(track.timestamped_objects)
                for timestamped_object in track.timestamped_objects:
                    # Stores bounding box of the face
                    box = timestamped_object.normalized_bounding_box
                    
                    # Finds bounding frames and time of the track
                    start_offset_seconds = self.__get_time(track.segment.start_time_offset)
                    end_offset_seconds = self.__get_time(track.segment.end_time_offset)
                    frames_in_track = int((end_offset_seconds - start_offset_seconds) * fps)
                    starting_frame = int(start_offset_seconds * fps)
                    
                    # Tracks whether we have identifed the person yet
                    detected_person = False
                    person_name = "Unknown"
                    
                    for frame in range(starting_frame, starting_frame + frames_in_track):
                        if not detected_person and (frame - starting_frame) % annotation_interval == 0:
                            # Gets the current frame and identifies all faces in it
                            curr_frame = self.__crop_face_frame(cap, frame, box)
                            identified_faces = self.recognition_model.identify_faces(curr_frame)
                            
                            # If we found someone, update tracking variables
                            if len(identified_faces) > 0:
                                detected_person = True
                                person_name = "TODO"
                        
                        # Appends this face to the frame's data
                        curr_faces = face_frames.get(frame, [])
                        new_face_info = [box, detected_person, person_name]
                        curr_faces.append(new_face_info)          
                        
        # Process each frame of the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_num in range(total_frames):
            # Make sure we can read the frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip if this frame has no faces
            if frame_num not in face_frames:
                continue
            
            # Extract identification information
            for face_info in face_frames[frame_num]:
                box = face_info[0]
                name = face_info[1]
                color = face_info[2] if (0, 255, 0) else (255, 0, 0)
                
                # Draws bounding box
                x1, y1, x2, y2 = int(box.left * width), int(box.top * height), int(box.right * width), int(box.bottom * height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Write name underneath bounding box
                cv2.addText(frame, name, (x1, y2))

            out.write(frame)  # Write the frame (with or without annotations)

        # Release everything when job is finished
        cap.release()
        out.release()
