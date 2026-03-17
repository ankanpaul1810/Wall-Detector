import streamlit as st
import os
import subprocess
from PIL import Image
import sys
import cv2
import numpy as np
import render_utils
from streamlit_image_comparison import image_comparison
from streamlit_drawable_canvas import st_canvas
import io

WALLPAPER_FOLDER = "wallpapers"
CAMERA_IMAGE_FILE = "captured_room.jpg"
MASK_OUTPUT_FILE = "wall_mask.png"  # AI generated mask
EDITED_MASK_FILE = "edited_mask.png"  # AI mask + user edits combined
FINAL_MASK_FILE = "final.png"  # Final mask for wallpaper application

st.set_page_config(page_title="AI Wallpaper Visualizer", layout="wide")
st.title("AI Wallpaper Visualizer")

if "room_path" not in st.session_state:
    st.session_state.room_path = None
if "selected_wallpaper_path" not in st.session_state:
    st.session_state.selected_wallpaper_path = None
if "mask_path" not in st.session_state:
    st.session_state.mask_path = None
if "room_image_cv" not in st.session_state:
    st.session_state.room_image_cv = None
if "texture_image_cv" not in st.session_state:
    st.session_state.texture_image_cv = None
if "camera_is_open" not in st.session_state:
    st.session_state.camera_is_open = False
if "show_editor" not in st.session_state:
    st.session_state.show_editor = False

st.subheader("Step 1: Upload Your Room Image")

col1, col2 = st.columns(2)
with col1:
    uploaded_room = st.file_uploader("Upload a room image", type=["jpg", "jpeg", "png"])
with col2:
    st.write("Capture now")
    if st.session_state.camera_is_open:
        if st.button("Close Camera"):
            st.session_state.camera_is_open = False
            st.rerun()
    else:
        if st.button("Open Camera"):
            st.session_state.camera_is_open = True
            st.rerun()

camera_capture = None
if st.session_state.camera_is_open:
    camera_capture = st.camera_input(
        "Capture from webcam", 
        label_visibility="collapsed"
    )

if uploaded_room:
    with open("uploaded_room.jpg", "wb") as f:
        f.write(uploaded_room.getbuffer())
    st.session_state.room_path = "uploaded_room.jpg"
    st.session_state.camera_is_open = False
elif camera_capture:
    with open(CAMERA_IMAGE_FILE, "wb") as f:
        f.write(camera_capture.getbuffer())
    st.session_state.room_path = CAMERA_IMAGE_FILE
    st.session_state.camera_is_open = False

if st.session_state.room_path:
    st.image(st.session_state.room_path, caption="Current Room", width=700)
else:
    st.info("Please upload or capture an image to begin.")

# wallpaper part
if st.session_state.room_path:
    st.markdown("---")
    st.subheader("Step 2: Select a Wallpaper")

    wallpapers = [
        os.path.join(WALLPAPER_FOLDER, f)
        for f in os.listdir(WALLPAPER_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    selected_wallpaper = None

    if wallpapers:
        st.write("Click on a wallpaper below to select it:")
        cols = st.columns(4)
        for i, wp_path in enumerate(wallpapers):
            with cols[i % 4]:
                wp_name = os.path.basename(wp_path)
                try:
                    img = Image.open(wp_path)
                    img.thumbnail((250, 250))
                    st.image(img, caption=wp_name, width=250)
                except Exception as e:
                    st.warning(f"Couldn't load {wp_name}: {e}")

                if st.button(f"Select {wp_name}", key=f"select_{i}"):
                    st.session_state.selected_wallpaper_path = wp_path
                    st.success(f"Selected: {wp_name}")

    else:
        st.warning("No wallpapers found. Please add some first.")

    if "selected_wallpaper_path" in st.session_state and st.session_state.selected_wallpaper_path:
        selected_wallpaper = st.session_state.selected_wallpaper_path
        st.markdown("---")
        st.subheader("Selected Wallpaper Preview")
        img = Image.open(selected_wallpaper)
        img.thumbnail((400, 400))
        st.image(img, caption=os.path.basename(selected_wallpaper))
    
    if st.session_state.room_path and st.session_state.selected_wallpaper_path:
        st.markdown("---")
        st.subheader("Step 3: Find Walls")
        
        multi_wall = st.checkbox("Apply to all detected walls (multi-wall mode)", value=True)

        if st.button("Find Walls (This takes ~1 minute)"):
            with st.spinner("Loading AI models and analyzing room... Please wait."):
                cmd = [
                    sys.executable, "run_pipeline2.py",
                    "--room", st.session_state.room_path,
                    "--output_mask", MASK_OUTPUT_FILE
                ]
                if not multi_wall:
                    cmd.append("--single_wall")
                
                print(f"Running command: {' '.join(cmd)}")
                try:
                    result = subprocess.run(
                        cmd, check=True, capture_output=True, text=True, timeout=300
                    )
                    print("Pipeline STDOUT:", result.stdout)
                    
                    if os.path.exists(MASK_OUTPUT_FILE):
                        st.success("Wall analysis complete!")
                        st.session_state.mask_path = MASK_OUTPUT_FILE
                        st.session_state.room_image_cv = cv2.imread(st.session_state.room_path)
                        st.session_state.texture_image_cv = cv2.imread(st.session_state.selected_wallpaper_path)
                        st.session_state.show_editor = False
                        
                        st.rerun()
                    else:
                        st.error("AI Pipeline ran but failed to create a mask file.")
                        st.code(result.stdout)
                        st.code(result.stderr)
                
                except subprocess.CalledProcessError as e:
                    st.error(f"Error finding walls (Exit Code {e.returncode}):")
                    st.code(e.stderr, language="bash")
                except subprocess.TimeoutExpired:
                    st.error("Error: The AI process timed out.")

if st.session_state.mask_path:
    # Mask Editor Section
    if not st.session_state.show_editor:
        st.markdown("---")
        st.subheader("Step 3.5: Review Detected Walls")
        
        mask_img = cv2.imread(st.session_state.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is not None and st.session_state.room_image_cv is not None:
            # Resize room to match mask dimensions
            mask_h, mask_w = mask_img.shape
            room_img = cv2.resize(st.session_state.room_image_cv, (mask_w, mask_h), interpolation=cv2.INTER_AREA)
            
            # Show preview with green overlay
            room_rgb = cv2.cvtColor(room_img, cv2.COLOR_BGR2RGB)
            overlay = room_rgb.copy()
            overlay[mask_img == 255] = [0, 255, 0]
            preview = cv2.addWeighted(room_rgb, 0.7, overlay, 0.3, 0)
            st.image(preview, caption="Detected Walls (Green)", width=700)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Edit Mask"):
                    st.session_state.show_editor = True
                    st.rerun()
            with col2:
                if st.button("Continue Without Editing"):
                    pass
    
    else:
        # Show editor
        st.markdown("---")
        st.subheader("Step 3.5: Edit Mask")
        st.info("WHITE = add walls | BLACK = remove walls | Editing directly on wall_mask.png")
        
        # ALWAYS load wall_mask.png to edit
        current_mask = cv2.imread(MASK_OUTPUT_FILE, cv2.IMREAD_GRAYSCALE)
        
        # Get dimensions
        orig_h, orig_w = current_mask.shape
        
        # Dynamic canvas scaling
        max_canvas_size = 1024
        scale = min(1.0, max_canvas_size / max(orig_w, orig_h))
        
        if scale * max(orig_w, orig_h) < 300:
            canvas_w, canvas_h = 600, 400
            scale = min(canvas_w / orig_w, canvas_h / orig_h)
        else:
            canvas_w = int(orig_w * scale)
            canvas_h = int(orig_h * scale)
        
        # Resize for canvas
        canvas_mask = cv2.resize(current_mask, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            brush_color = st.radio("Brush", ["White", "Black"])
            base_brush_size = st.slider("Size", 5, 50, 15)
            brush_size = max(1, int(base_brush_size * scale))
            
            st.write(f"Canvas: {canvas_w}x{canvas_h}")
            st.write(f"Original: {orig_w}x{orig_h}")
            
            if st.button("Reset Edits"):
                # Reload to discard edits
                st.rerun()
                
            if st.button("Done Editing"):
                # The canvas has wall_mask.png + your edits combined
                # This was saved in real-time to a temp location
                # Now we finalize it as edited_mask.png
                
                if os.path.exists("temp_canvas_mask.png"):
                    # Read what's currently on canvas (AI mask + edits)
                    final_edited = cv2.imread("temp_canvas_mask.png", cv2.IMREAD_GRAYSCALE)
                    
                    # Save as edited_mask.png
                    cv2.imwrite(EDITED_MASK_FILE, final_edited)
                    
                    # Use edited_mask.png for wallpaper
                    st.session_state.mask_path = EDITED_MASK_FILE
                    
                    # Resize room image
                    mask_h, mask_w = final_edited.shape
                    st.session_state.room_image_cv = cv2.resize(
                        st.session_state.room_image_cv,
                        (mask_w, mask_h),
                        interpolation=cv2.INTER_AREA
                    )
                    
                    st.success("Edits saved to edited_mask.png!")
                
                st.session_state.show_editor = False
                st.rerun()
        
        with col1:
            color_hex = "#FFFFFF" if brush_color == "White" else "#000000"
            
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=brush_size,
                stroke_color=color_hex,
                background_image=Image.fromarray(canvas_mask),
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode="freedraw",
                key="canvas"
            )
            
            # Save canvas by MERGING AI mask + your strokes
            if canvas_result.image_data is not None:
                # Get what's on canvas (should have background + strokes)
                canvas_rgb = canvas_result.image_data[:, :, :3].astype(np.uint8)
                canvas_gray = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2GRAY)
                _, canvas_binary = cv2.threshold(canvas_gray, 127, 255, cv2.THRESH_BINARY)
                
                # Upscale canvas to original size
                canvas_full = cv2.resize(canvas_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                # IMPORTANT: Merge with original AI mask to ensure we keep AI detection
                ai_mask_original = cv2.imread(MASK_OUTPUT_FILE, cv2.IMREAD_GRAYSCALE)
                
                # Combine: Take maximum of both (union of AI mask and your edits)
                combined = cv2.bitwise_or(ai_mask_original, canvas_full)
                
                # Save combined result
                cv2.imwrite("temp_canvas_mask.png", combined)
    
    # Step 4: Live Adjustments
    if not st.session_state.show_editor:
        st.markdown("---")
        st.subheader("Step 4: Live Adjustments")
        st.info("All adjustments below are real-time.")
        
        mask_image = cv2.imread(st.session_state.mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask_image is None:
            st.error("Failed to load the wall mask.")
            st.session_state.mask_path = None 
        elif st.session_state.room_image_cv is None or st.session_state.texture_image_cv is None:
            st.warning("Image data lost. Re-loading images...")
            st.session_state.room_image_cv = cv2.imread(st.session_state.room_path)
            st.session_state.texture_image_cv = cv2.imread(st.session_state.selected_wallpaper_path)
            if st.session_state.room_image_cv is None or st.session_state.texture_image_cv is None:
                st.error("Failed to reload images. Please start over from Step 1.")
                st.session_state.clear() 
                st.stop()
        else:
            mask_h, mask_w = mask_image.shape[:2]
            room_h, room_w, _ = st.session_state.room_image_cv.shape
            if (room_h, room_w) != (mask_h, mask_w):
                print(f"Resizing source image from {room_w}x{room_h} to match mask {mask_w}x{mask_h}")
                st.session_state.room_image_cv = cv2.resize(
                    st.session_state.room_image_cv, 
                    (mask_w, mask_h), 
                    interpolation=cv2.INTER_AREA
                )
            
            zoom = st.slider("Texture Zoom (Pattern Size)", 0.2, 3.0, 1.0, 0.1)
            brightness = st.slider("Brightness", 0, 100, 50, 1)
            contrast = st.slider("Contrast", 0, 100, 50, 1)
            saturation = st.slider("Saturation", 0, 100, 50, 1)

            final_image_bgr = render_utils.apply_design(
                st.session_state.room_image_cv,
                st.session_state.texture_image_cv,
                mask_image,
                zoom,
                brightness,
                contrast,
                saturation
            )
            
            final_image_rgb = cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB)
            room_image_rgb = cv2.cvtColor(st.session_state.room_image_cv, cv2.COLOR_BGR2RGB)
            
            st.markdown("---")
            st.subheader("Final Preview")
            st.image(final_image_rgb, caption="Adjusted Preview", width=700)
            
            # Download button
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                final_pil = Image.fromarray(final_image_rgb)
                buf = io.BytesIO()
                final_pil.save(buf, format="PNG")
                st.download_button(
                    label="Download Result",
                    data=buf.getvalue(),
                    file_name="wallpaper_result.png",
                    mime="image/png"
                )

            st.markdown("---")
            st.subheader("Before / After Comparison")
            image_comparison(
                img1=room_image_rgb,
                img2=final_image_rgb,
                label1="Before",
                label2="After"
            )