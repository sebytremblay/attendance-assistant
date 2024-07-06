import io

from google.cloud import videointelligence_v1 as videointelligence

class FaceDetector:
    def __init__(self):
        pass
    
    def detect_faces(self, video_path):
        """Detects faces in a video from a local file.

        Returns:
            AnnotatedVideoResponse: The response containing the face detection annotations.
        """
        client = videointelligence.VideoIntelligenceServiceClient()

        with io.open(video_path, "rb") as f:
            input_content = f.read()

        # Configure the request
        config = videointelligence.FaceDetectionConfig(
            model="builtin/latest",
            include_bounding_boxes=True, 
            include_attributes=True
        )
        context = videointelligence.VideoContext(face_detection_config=config)

        # Start the asynchronous request
        print("\nProcessing video for face detection annotations.")
        operation = client.annotate_video(
            request={
                "features": [videointelligence.Feature.FACE_DETECTION],
                "input_content": input_content,
                "video_context": context,
            }
        )

        result = operation.result(timeout=300)
        print("\nFinished processing.")

        # Return the first result, because a single video was processed.
        return result.annotation_results[0]