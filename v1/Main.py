import io
import cv2
import os
import shutil

from google.cloud import videointelligence_v1 as videointelligence

# Define the output folder as a global variable
OUTPUT_FOLDER = "./output"
PURGE_OUTPUT_ON_LAUNCH = True

def clear_output_folder():
    """Clear all files in the output folder."""
    if not PURGE_OUTPUT_ON_LAUNCH:
        return
    
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_time(time_offset):
    """Calculate the time in seconds from a time_offset object.

    Args:
        time_offset (TimeOffset): The time offset object from video annotation.

    Returns:
        float: Time in seconds.
    """
    return time_offset.seconds + time_offset.microseconds / 1e6

def capture_and_save_frame(cap, frame_num, box, fps, descriptor):
    """Capture a specific frame from a video and save the detected face.

    Args:
        cap (cv2.VideoCapture): Video capture object.
        frame_num (int): Frame number to capture.
        box (NormalizedBoundingBox): Bounding box for the face in the frame.
        fps (float): Frames per second of the video.
        descriptor (str): Descriptor for the frame (e.g., 'first', 'middle', 'last').

    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to capture {descriptor} frame at timestamp:", frame_num / fps)
        return

    face_frame = crop_face_frame(frame, box)
    face_frame_filename = f"./output/face_frame_{descriptor}_at_{frame_num}_box_{box.left}_{box.top}_{box.right}_{box.bottom}.jpg"
    cv2.imwrite(face_frame_filename, face_frame)
    print(f"Saved {face_frame_filename}")

def crop_face_frame(frame, box):
    """Crop the face frame from the given frame based on the bounding box.

    Args:
        frame (numpy.ndarray): The frame from which to crop the face.
        box (NormalizedBoundingBox): The bounding box of the face in the frame.

    Returns:
        numpy.ndarray: The cropped face frame.
    """
    h, w = frame.shape[:2]
    border_size = 25
    x1, y1, x2, y2 = int(box.left * w), int(box.top * h), int(box.right * w), int(box.bottom * h)

    x1_border = max(x1 - border_size, 0)
    y1_border = max(y1 - border_size, 0)
    x2_border = min(x2 + border_size, w - 1)
    y2_border = min(y2 + border_size, h - 1)

    return frame[y1_border:y2_border, x1_border:x2_border]

def save_detected_faces(video_path, fps):
    """Detect and save the first, middle, and last frames of faces in a video.

    Args:
        video_path (str): Path to the video file.
        fps (float): Frames per second of the video.
    """
    annotation_result = detect_faces(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    for annotation in annotation_result.face_detection_annotations:
        for track in annotation.tracks:
            start_offset_seconds = get_time(track.segment.start_time_offset)
            end_offset_seconds = get_time(track.segment.end_time_offset)
            midpoint_offset_seconds = (start_offset_seconds + end_offset_seconds) / 2

            for timestamp, descriptor in [
                (start_offset_seconds, 'first'), 
                (midpoint_offset_seconds, 'middle'), 
                (end_offset_seconds, 'last')
            ]:
                 # Find the timestamped object closest to the descriptor
                closest_timestamped_object = min(track.timestamped_objects,
                                                key=lambda obj: abs(
                                                    get_time(obj.time_offset) - timestamp))
                
                # Calculate frame number based on the midpoint time offset of the object
                frame_num = int(get_time(closest_timestamped_object.time_offset) * fps)

                # Use the bounding box from the closest timestamped object
                box = closest_timestamped_object.normalized_bounding_box
                
                # Saves the cropped image
                capture_and_save_frame(cap, frame_num, box, fps, descriptor)

    cap.release()

def detect_faces(local_file_path):
    """Detects faces in a video from a local file.

    Args:
        local_file_path (str): Path to the local video file.

    Returns:
        AnnotatedVideoResponse: The response containing the face detection annotations.
    """
    client = videointelligence.VideoIntelligenceServiceClient()

    with io.open(local_file_path, "rb") as f:
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

if __name__ == "__main__":
    clear_output_folder()
    save_detected_faces("./examples/test3.MOV", 29.99)