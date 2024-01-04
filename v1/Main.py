from Controller import Controller

if __name__ == "__main__":
    KEY = "723831a338d34248846f5d53e4896dae"
    ENDPOINT = "https://delt-facial-recognition-v1.cognitiveservices.azure.com/"
    TRAINING_DATA_PATH = "./training_data"
    
    controller = Controller(KEY, ENDPOINT)
    controller.train_model(TRAINING_DATA_PATH)
    
    video_path = "./examples/test3.MOV"
    fps = 29.99
    controller.analyze_video(video_path, fps)