import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.title("Canvas Test")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=10,
    stroke_color="black",
    background_color="gray",
    update_streamlit=True,
    height=400,
    width=600,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
