import streamlit as st
from streamlit_image_comparison import image_comparison
from PIL import Image

# Set Streamlit page configuration
st.set_page_config(page_title="Image Comparison Example", layout="centered")

st.title("Image Comparison Demo")

# Load images (replace with your actual image paths or URLs)
# For demonstration, let's assume 'image1.jpg' and 'image2.jpg' exist in the same directory
# Or you can use URLs, PIL Image objects, or OpenCV images
img1_path = "C:\\Users\\Ankan\\Desktop\\wall\\test_result.jpg"
img2_path = "C:\\Users\\Ankan\\Desktop\\wall\\test_result1.jpg"

# Example of creating dummy images for demonstration if actual images are not available
try:
    # Attempt to open local images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
except FileNotFoundError:
    st.warning("Image files not found. Using placeholder images for demonstration.")
    # Create simple placeholder images if files don't exist
    from PIL import ImageDraw
    img1 = Image.new('RGB', (400, 300), color = 'red')
    draw1 = ImageDraw.Draw(img1)
    draw1.text((50,150), "Before", fill=(255,255,255))

    img2 = Image.new('RGB', (400, 300), color = 'blue')
    draw2 = ImageDraw.Draw(img2)
    draw2.text((50,150), "After", fill=(255,255,255))


# Render the image comparison slider
image_comparison(
    img1=img1,
    img2=img2,
    label1="Original Image",
    label2="Processed Image",
    width=700,
    starting_position=50, # Initial position of the slider (0-100)
    show_labels=True,
    make_responsive=True,
    in_memory=True, # Set to True if passing PIL or OpenCV images directly
)

st.write("Move the slider to compare the two images.")