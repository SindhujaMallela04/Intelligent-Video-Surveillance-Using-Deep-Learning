# display.py

import cv2
import numpy as np
from preprocessing import rec_frame_display, display_text_fnc

def display_prediction(frame, predictions, roi):
    """
    Draws the predictions and overlays on the frame.
    
    Args:
        frame: Original BGR frame.
        predictions: A dictionary with prediction data (e.g., label and score).
        roi: Region of Interest coordinates for overlay.
    
    Returns:
        Annotated frame.
    """
    frame = rec_frame_display(frame, roi)

    label = predictions.get("label", "Unknown")
    score = predictions.get("score", 0)

    display_text = f"Prediction: {label} ({score*100:.2f}%)"
    frame = display_text_fnc(frame, display_text, index=0)

    return frame

def show_video(title, frame, wait_time=1):
    """
    Display a single video frame with OpenCV.
    
    Args:
        title: Window title.
        frame: The frame to display.
        wait_time: Time in milliseconds to wait between frames.
    """
    cv2.imshow(title, frame)
    key = cv2.waitKey(wait_time)
    return key

def overlay_status(frame, status_text):
    """
    Overlay system status on the top-left of the frame.
    
    Args:
        frame: Video frame.
        status_text: Text to overlay.
        
    Returns:
        Frame with status overlay.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 255, 255)
    thickness = 2
    position = (10, 30)

    cv2.putText(frame, status_text, position, font, font_scale, font_color, thickness)
    return frame
