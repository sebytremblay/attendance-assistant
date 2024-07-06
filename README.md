# attendance-assistant
An application to record club members' attendance at meetings. Leverages GCP's facial detection model to detect faces and passes to Azure's facial recognition service to identify them.

## Project Structure

- FaceRecognizer Class
    - Train model
    - Add new people
    - Classify a face
- FaceDetection Class
    - Extract face tracks
- VideoAnalytics Class
    - Iterate over all face tracks
    - Run face detection every X frames until face is classified


- List of tracks where a track contains a bounding box and timestamp
- Iterate through track. If we have not detected a face yet, run facial detection every X steps.
