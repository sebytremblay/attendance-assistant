from FaceDetection import FaceDetection

if __name__ == "__main__":
    video_path = "./examples/test1.MOV"
    fps = 29.99
    
    controller = FaceDetection(video_path, fps)
    controller.save_detected_faces()