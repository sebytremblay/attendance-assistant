import credentials as credentials

from v2.Controller import Controller

if __name__ == "__main__":
    # Defines request arguments
    video_path = "./examples/test2.MOV"
    frame_interval = 30*5
    train_model = True
    training_data_path = "./training_data/Delt"
    person_group = ""
    
    # Analyzes the provided video
    controller = Controller(credentials.API_KEY, credentials.API_ENDPOINT, person_group)
    controller.analyze_video(
        video_path, frame_interval, 
        train_model, training_data_path)