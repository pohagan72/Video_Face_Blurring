import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import os

# Load YOLO model
model = YOLO("yolov8n.pt")  # You can replace this with a custom face detection model if needed

# Streamlit UI
st.title("Video Face Blur")
st.write("Upload a video to blur faces")

# Input video file
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        st.stop()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a temporary output video file
    output_path = "output_blurred.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Process each frame
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using YOLO
        results = model(frame)
        for result in results:
            for box in result.boxes:
                if result.names[int(box.cls)] == "person":  # Check if the detected object is a person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates

                    # Blur the face region
                    face_region = frame[y1:y2, x1:x2]
                    blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                    frame[y1:y2, x1:x2] = blurred_face

        # Write the frame to the output video
        out.write(frame)

        # Update progress
        progress = (frame_idx + 1) / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_idx + 1} of {total_frames}")

    # Release resources
    cap.release()
    out.release()

    # Display the output video
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
