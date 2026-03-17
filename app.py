import streamlit as st
import os
import subprocess
from PIL import Image
import sys

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
WALLPAPER_FOLDER = "wallpapers"
OUTPUT_FILE = "room_with_design_FINAL6.jpg"
CAMERA_IMAGE_FILE = "captured_room.jpg"
# ----------------------------------------------------------

st.set_page_config(page_title="AI Wallpaper Visualizer", layout="wide")
st.title("AI Wallpaper Visualizer")

# --------------------------------------------
# ROOM IMAGE INPUT: Upload OR Camera Capture
# --------------------------------------------
st.subheader("Upload or Capture Your Room")

# --- FIX 1: Initialize session_state keys ---
if "camera_open" not in st.session_state:
    st.session_state.camera_open = False
# This key will store the path to the active room image
if "room_path" not in st.session_state:
    st.session_state.room_path = None

# Use columns for layout
col1, col2 = st.columns(2)

with col1:
    uploaded_room = st.file_uploader("1. Upload a room image", type=["jpg", "jpeg", "png"])

with col2:
    st.write("2. Or use your camera:")
    if st.button("Open Camera"):
        st.session_state.camera_open = True

# Conditionally display the camera input
camera_capture = None
if st.session_state.camera_open:
    camera_capture = st.camera_input("Capture directly from your webcam. Press the button again to close.", key="camera")

# --- FIX 2: Update session_state when an image is ready ---
if uploaded_room:
    # Save the uploaded file
    with open("uploaded_room.jpg", "wb") as f:
        f.write(uploaded_room.getbuffer())
    # Save the path to session_state
    st.session_state.room_path = "uploaded_room.jpg"
    # If user uploads, we can hide the camera input
    st.session_state.camera_open = False

elif camera_capture:
    # Save the captured file
    with open(CAMERA_IMAGE_FILE, "wb") as f:
        f.write(camera_capture.getbuffer())
    # Save the path to session_state
    st.session_state.room_path = CAMERA_IMAGE_FILE
    # If user takes a picture, we can hide the camera input
    st.session_state.camera_open = False

# --- FIX 3: Always display the image from session_state ---
if st.session_state.room_path:
    # Display the currently active room image
    st.image(st.session_state.room_path, caption="Current Room", use_container_width=True)


# --------------------------------------------
# WALLPAPER SELECTION (Grid View)
# --------------------------------------------
st.markdown("---")
st.subheader("Select a Wallpaper")

wallpapers = [
    os.path.join(WALLPAPER_FOLDER, f)
    for f in os.listdir(WALLPAPER_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", "png"))
]

selected_wallpaper = None

if wallpapers:
    st.write("Click on a wallpaper below to select it:")
    # Adjust columns based on the number of wallpapers, e..g., 5 columns
    cols = st.columns(5) 
    for i, wp_path in enumerate(wallpapers):
        with cols[i % 5]:
            wp_name = os.path.basename(wp_path)
            try:
                img = Image.open(wp_path)
                img.thumbnail((250, 250))
                st.image(img, caption=wp_name, use_container_width=True)
            except Exception as e:
                st.warning(f"Couldn't load {wp_name}: {e}")

            if st.button(f"Select", key=f"select_{i}"):
                st.session_state["selected_wallpaper"] = wp_path
                st.success(f"Selected: {wp_name}")

else:
    st.warning(f"No wallpapers found in '{WALLPAPER_FOLDER}' folder. Please add some first.")

if "selected_wallpaper" in st.session_state:
    selected_wallpaper = st.session_state["selected_wallpaper"]
    st.markdown("---")
    st.subheader("Selected Wallpaper Preview")
    img = Image.open(selected_wallpaper)
    img.thumbnail((400, 400))
    st.image(img, caption=os.path.basename(selected_wallpaper))

# --------------------------------------------
# SETTINGS: LIGHTING, MATTE, MULTI-WALL MODE
# --------------------------------------------
st.markdown("---")
st.subheader("Adjustment Controls")

lighting_strength = st.slider("Lighting Adjustment", 0.0, 1.0, 0.2, 0.05)
matte_strength = st.slider("Matte / Gloss Effect", 0.0, 1.0, 0.3, 0.05)
multi_wall = st.checkbox("Apply to all detected walls (multi-wall mode)", value=True)

# --------------------------------------------
# RUN PIPELINE BUTTON
# --------------------------------------------
if st.button("Generate Preview"):
    
    # --- FIX 4: Read the room_path from session_state ---
    room_path = st.session_state.room_path
    
    if not room_path:
        st.error("Please upload or capture a room image first.")
    elif not selected_wallpaper:
        st.error("Please select a wallpaper.")
    else:
        with st.spinner("Processing... this may take up to a minute."):

            cmd = [
                sys.executable, "run_pipeline1.py",
                "--room", room_path,
                "--design", selected_wallpaper,
                "--output", OUTPUT_FILE,
                "--lighting", str(lighting_strength),
                "--matte", str(matte_strength)
            ]

            if not multi_wall:
                cmd.append("--single_wall")
            
            print(f"Running command: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd, 
                    check=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=300 
                )
                print("Pipeline STDOUT:", result.stdout)
                if os.path.exists(OUTPUT_FILE):
                    st.success("Wallpaper applied successfully!")
                    st.image(OUTPUT_FILE, caption="Preview Result", use_container_width=True)
                else:
                    st.error("Pipeline ran, but failed to generate the output image.")
                    st.text(result.stdout)
                    st.text(result.stderr)
                    
            except subprocess.CalledProcessError as e:
                st.error(f"Error running pipeline (Exit Code {e.returncode}):")
                st.code(e.stderr, language="bash")
                print("Pipeline STDERR:", e.stderr)
            except subprocess.TimeoutExpired as e:
                st.error("Error: The process timed out. This may be due to a very large image or slow model loading.")
                print("Pipeline TIMEOUT:", e.stderr)

