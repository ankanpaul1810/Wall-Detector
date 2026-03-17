import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import render_utils
import os

EDITED_MASK_FILE = "edited_mask.png"

def show_mask_editor(state):
    st.markdown("---")
    st.subheader("🧩 Step 3.5: Edit Mask")

    # --- Load current mask ---
    original_mask = cv2.imread(state.mask_path, cv2.IMREAD_GRAYSCALE)
    if original_mask is None:
        st.warning("Mask not found. Please rerun 'Find Walls'.")
        st.stop()

    if os.path.exists(EDITED_MASK_FILE):
        current_mask = cv2.imread(EDITED_MASK_FILE, cv2.IMREAD_GRAYSCALE)
        if current_mask is None:
            current_mask = original_mask.copy()
    else:
        current_mask = original_mask.copy()

    # --- Sidebar controls ---
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("### 🎨 Tools")
        brush_color = st.radio("Brush", ["White", "Black"])
        brush_size = st.slider("Brush Size", 5, 100, 20)

        st.markdown("### 🔍 Zoom")
        zoom_pct = st.slider("Zoom Level (%)", 25, 100, 100, step=5)
        zoom_scale = zoom_pct / 100.0

        st.markdown("### 👁️ View Mode")
        view_mode = st.radio("Mode", ["Mask View", "Wallpaper View"], horizontal=True)

        if st.button("🔄 Reset to Original"):
            if os.path.exists(EDITED_MASK_FILE):
                os.remove(EDITED_MASK_FILE)
            state.mask_path = "wall_mask.png"
            st.success("✅ Restored AI-generated mask.")
            st.rerun()

        if st.button("✅ Done Editing"):
            if os.path.exists(EDITED_MASK_FILE):
                state.mask_path = EDITED_MASK_FILE
            state.show_editor = False
            st.rerun()

    # --- Main canvas area ---
    with col1:
        mask_h, mask_w = current_mask.shape[:2]

        # ✅ Apply zoom scaling
        display_h = int(mask_h * zoom_scale)
        display_w = int(mask_w * zoom_scale)
        display_mask = cv2.resize(current_mask, (display_w, display_h), interpolation=cv2.INTER_NEAREST)

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=brush_size,
            stroke_color="#FFFFFF" if brush_color == "White" else "#000000",
            background_image=Image.fromarray(display_mask),
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode="freedraw",
            key="mask_canvas_fullres",
        )

        # --- Process edits ---
        if canvas_result.image_data is not None:
            canvas_img = (canvas_result.image_data * 255).astype(np.uint8)
            canvas_rgb = canvas_img[:, :, :3]
            ch, cw = canvas_rgb.shape[:2]

            bg_display = cv2.resize(display_mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
            bg_rgb = cv2.cvtColor(bg_display, cv2.COLOR_GRAY2BGR)
            diff = np.any(canvas_rgb != bg_rgb, axis=2)

            if np.any(diff):
                luminance = (
                    canvas_rgb[:, :, 0]*0.2126 +
                    canvas_rgb[:, :, 1]*0.7152 +
                    canvas_rgb[:, :, 2]*0.0722
                ).astype(np.uint8)

                white_mask = np.zeros((ch, cw), np.uint8)
                black_mask = np.zeros((ch, cw), np.uint8)
                white_mask[(diff) & (luminance >= 200)] = 255
                black_mask[(diff) & (luminance <= 50)] = 255

                # ✅ Rescale painted masks back to original resolution
                white_full = cv2.resize(white_mask, (mask_w, mask_h), interpolation=cv2.INTER_NEAREST)
                black_full = cv2.resize(black_mask, (mask_w, mask_h), interpolation=cv2.INTER_NEAREST)

                combined = current_mask.copy()
                combined[white_full == 255] = 255
                combined[black_full == 255] = 0
                cv2.imwrite(EDITED_MASK_FILE, combined)

                # --- View mode rendering ---
                if view_mode == "Mask View":
                    room = state.room_image_cv
                    if room is not None:
                        rh, rw = combined.shape
                        room_resized = cv2.resize(room, (rw, rh))
                        room_rgb = cv2.cvtColor(room_resized, cv2.COLOR_BGR2RGB)
                        overlay = room_rgb.copy()
                        overlay[combined == 255] = [0, 255, 0]
                        preview = cv2.addWeighted(room_rgb, 0.7, overlay, 0.3, 0)
                        st.image(preview, caption=f"🟩 Mask Overlay ({zoom_pct}%)", width=700)
                    else:
                        st.image(combined, caption="Mask View", width=700)
                else:
                    room = state.room_image_cv
                    texture = state.texture_image_cv
                    if room is not None and texture is not None:
                        rh, rw = combined.shape
                        room_r = cv2.resize(room, (rw, rh))
                        tex_r = cv2.resize(texture, (rw, rh))
                        wallpaper = render_utils.apply_design(room_r, tex_r, combined, 1.0, 50, 50, 50)
                        st.image(cv2.cvtColor(wallpaper, cv2.COLOR_BGR2RGB),
                                 caption=f"🎨 Wallpaper View ({zoom_pct}%)", width=700)
            else:
                st.image(current_mask, caption=f"No edits yet ({zoom_pct}%)", width=700)

# -------------------------------------------------------
# 🧠 Standalone Debug Mode
# -------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Mask Editor Debug", layout="wide")
    st.title("🧪 Mask Editor Debug Mode")

    class DummyState:
        def __init__(self):
            self.mask_path = "debug_mask.png"
            self.room_image_cv = None
            self.texture_image_cv = None
            self.show_editor = True

    # 🔧 Generate dummy files if missing
    if not os.path.exists("debug_mask.png"):
        mask = np.zeros((1080, 1920), np.uint8)
        mask[200:800, 400:1500] = 255
        cv2.imwrite("debug_mask.png", mask)
    if not os.path.exists("debug_room.jpg"):
        dummy_room = np.full((1080, 1920, 3), 180, np.uint8)
        cv2.imwrite("debug_room.jpg", dummy_room)
    if not os.path.exists("debug_texture.jpg"):
        dummy_tex = np.zeros((1080, 1920, 3), np.uint8)
        for i in range(0, 1920, 40):
            cv2.line(dummy_tex, (i, 0), (i, 1080), (80, 140, 250), 4)
        for j in range(0, 1080, 40):
            cv2.line(dummy_tex, (0, j), (1920, j), (250, 200, 100), 3)
        cv2.imwrite("debug_texture.jpg", dummy_tex)

    # 🔁 Load dummy data
    state = DummyState()
    state.room_image_cv = cv2.imread("debug_room.jpg")
    state.texture_image_cv = cv2.imread("debug_texture.jpg")

    # ✅ Run editor
    show_mask_editor(state)
