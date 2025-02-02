# Import necessary libraries
import cv2  # OpenCV for video processing
import streamlit as st  # Streamlit for creating web apps
from ultralytics import YOLO  # YOLO model for object detection
import tempfile  # For creating temporary files
import os  # For file operations

# Load YOLO model for object detection
# Here, "yolov8n.pt" is a pre-trained model file; you can replace it with a custom model if needed
model = YOLO("yolov8n.pt")

# Streamlit UI setup
st.title("Video Face Blur")  # Set the title of the app
st.write("Upload a video to blur faces")  # Provide instructions to the user

# Input video file
# Allows users to upload a video file (mp4, avi, mov formats)
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:  # Check if a file has been uploaded
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())  # Write the uploaded file's content to the temp file
        video_path = tfile.name  # Get the path of the temporary file

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")  # Show an error message if the video cannot be opened
        st.stop()  # Stop the app

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the width of each frame
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the height of each frame
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames

    # Create a temporary output video file
    output_path = "output_blurred.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    # Initialize a progress bar to show processing status
    progress_bar = st.progress(0)
    status_text = st.empty()  # Placeholder for status text

    # Process each frame of the video
    for frame_idx in range(total_frames):
        ret, frame = cap.read()  # Read the next frame
        if not ret:
            break  # Exit the loop if no more frames are available

        # Detect faces in the frame using the YOLO model
        results = model(frame)
        for result in results:
            for box in result.boxes:
                if result.names[int(box.cls)] == "person":  # Check if the detected object is a person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates

                    # Blur the detected face region
                    face_region = frame[y1:y2, x1:x2]  # Extract the face region from the frame
                    blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)  # Apply Gaussian blur
                    frame[y1:y2, x1:x2] = blurred_face  # Replace the original face region with the blurred one

        # Write the processed frame to the output video
        out.write(frame)

        # Update the progress bar and status text
        progress = (frame_idx + 1) / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_idx + 1} of {total_frames}")

    # Release resources (video capture and writer)
    cap.release()
    out.release()

    # Display the output video in the Streamlit app
    st.video(output_path)

    # Provide a download link for the output video
    with open(output_path, "rb") as f:
        st.download_button(
            label="Download Blurred Video",
            data=f,
            file_name="output_blurred.mp4",
            mime="video/mp4",
        )

    # Clean up temporary files
    try:
        os.unlink(video_path)  # Delete the temporary input video file
        os.unlink(output_path)  # Delete the temporary output video file
    except PermissionError as e:
        st.warning(f"Could not delete temporary files due to a permission error: {e}")
