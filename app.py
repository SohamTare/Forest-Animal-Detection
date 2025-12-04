import streamlit as st
import cv2
import numpy as np
import tempfile
import os

from forest_animal_detector1 import detect_animals, draw_counts_on_frame, build_info_panel

st.set_page_config(page_title="Forest Animal Detection", layout="wide")

st.title("🌳 Forest Animal Detection with Species Info")
st.write("Upload an image or video and let AI detect animals with species info!")

uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    
    # Save upload to a temp file
    temp_input = tempfile.NamedTemporaryFile(delete=False)
    temp_input.write(uploaded_file.read())
    temp_input.close()

    if "image" in file_type:
        st.subheader("Processing Image...")
        
        frame = cv2.imread(temp_input.name)
        frame, detected_species, species_counts = detect_animals(frame)
        frame = draw_counts_on_frame(frame, species_counts)
        panel = build_info_panel(frame.shape[0], detected_species)
        combined = np.concatenate((frame, panel), axis=1)

        st.image(combined, channels="BGR")

        save_name = temp_input.name + "_annotated.png"
        cv2.imwrite(save_name, combined)

        with open(save_name, "rb") as f:
            st.download_button("⬇ Download Annotated Image", f, "annotated_image.png")

    elif "video" in file_type:
        st.subheader("Processing Video... This may take some time ⏳")

        cap = cv2.VideoCapture(temp_input.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_w = width + 380
        out_h = height

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out = cv2.VideoWriter(temp_output.name, fourcc, fps, (out_w, out_h))

        progress = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            frame, detected_species, species_counts = detect_animals(frame)
            frame = draw_counts_on_frame(frame, species_counts)
            panel = build_info_panel(frame.shape[0], detected_species)
            combined = np.concatenate((frame, panel), axis=1)

            out.write(combined)
            progress.progress(frame_no / frame_count)

        cap.release()
        out.release()
        st.success("Video Processing Completed 🎬")

        with open(temp_output.name, "rb") as f:
            st.download_button("⬇ Download Annotated Video", f, "annotated_video.mp4")
