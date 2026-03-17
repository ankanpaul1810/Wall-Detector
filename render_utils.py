import cv2
import numpy as np

def make_texture_seamless(texture, detail_boost=True):
    """
    Makes a texture seamless by blending its opposite edges.
    """
    h, w = texture.shape[:2]
    if h == 0 or w == 0:
        return texture

    texture = texture.astype(np.float32)

    blend_width = max(1, w // 6)
    blend_height = max(1, h // 6)

    # Left-Right blend
    if w > blend_width * 2:
        left = texture[:, :blend_width]
        right = texture[:, -blend_width:]
        mix_lr = cv2.addWeighted(left, 0.8, cv2.flip(right, 1), 0.2, 0)
        texture[:, :blend_width] = mix_lr
        texture[:, -blend_width:] = cv2.flip(mix_lr, 1)

    # Top-Bottom blend
    if h > blend_height * 2:
        top = texture[:blend_height, :]
        bottom = texture[-blend_height:, :]
        mix_tb = cv2.addWeighted(top, 0.8, cv2.flip(bottom, 0), 0.2, 0)
        texture[:blend_height, :] = mix_tb
        texture[-blend_height:, :] = cv2.flip(mix_tb, 0)

    if detail_boost:
        blur = cv2.GaussianBlur(texture, (3, 3), 0)
        texture = cv2.addWeighted(texture, 1.2, blur, -0.2, 0)

    return np.clip(texture, 0, 255).astype(np.uint8)


def tile_texture(texture_image, room_w, room_h, zoom=1.0):
    """
    Tiles a seamless texture to fit the room dimensions, with a zoom factor.
    """
    if texture_image is None or room_w == 0 or room_h == 0:
        return np.zeros((room_h, room_w, 3), dtype=np.uint8)

    tex_h, tex_w = texture_image.shape[:2]
    if tex_h == 0 or tex_w == 0:
        return np.zeros((room_h, room_w, 3), dtype=np.uint8)

    zoomed_w = max(1, int(tex_w * zoom))
    zoomed_h = max(1, int(tex_h * zoom))

    zoomed_texture = cv2.resize(texture_image, (zoomed_w, zoomed_h), interpolation=cv2.INTER_LINEAR)
    if zoomed_texture.ndim == 2:
        zoomed_texture = cv2.cvtColor(zoomed_texture, cv2.COLOR_GRAY2BGR)

    tiled_seamless = make_texture_seamless(zoomed_texture.copy(), detail_boost=True)
    if tiled_seamless.shape[0] == 0 or tiled_seamless.shape[1] == 0:
        return np.zeros((room_h, room_w, 3), dtype=np.uint8)

    rep_x = max(1, int(np.ceil(room_w / tiled_seamless.shape[1])))
    rep_y = max(1, int(np.ceil(room_h / tiled_seamless.shape[0])))

    tiled = np.tile(tiled_seamless, (rep_y, rep_x, 1))
    return tiled[:room_h, :room_w]


# ----------------------------------------------------------
# REAL-TIME ADJUSTMENTS (Fast)
# ----------------------------------------------------------
def apply_brightness_contrast(image, brightness, contrast):
    b_val = (brightness - 50) * 2.54
    c_val = (contrast / 50.0)
    adjusted = cv2.convertScaleAbs(image, alpha=c_val, beta=b_val)
    return adjusted


def apply_saturation(image, saturation):
    s_val = (saturation / 50.0)
    if s_val == 1.0:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * s_val, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge([h, s, v])
    adjusted = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return adjusted


def apply_lighting_and_color(room_image, tiled_texture):
    """
    Preserves original wall lighting and shadows for realism.
    """
    room_lab = cv2.cvtColor(room_image, cv2.COLOR_BGR2LAB)
    texture_lab = cv2.cvtColor(tiled_texture, cv2.COLOR_BGR2LAB)

    room_L, room_A, room_B = cv2.split(room_lab)
    tex_L, tex_A, tex_B = cv2.split(texture_lab)

    room_L = room_L.astype(np.float32)
    tex_L = tex_L.astype(np.float32)

    room_L_mean = np.mean(room_L)
    tex_L_mean = np.mean(tex_L)
    tex_L_adjusted = tex_L - tex_L_mean + room_L_mean
    blended_L = cv2.addWeighted(room_L, 0.7, tex_L_adjusted, 0.3, 0)
    blended_L = np.clip(blended_L, 0, 255).astype(np.uint8)

    lit_design_lab = cv2.merge([blended_L, tex_A, tex_B])
    return cv2.cvtColor(lit_design_lab, cv2.COLOR_LAB2BGR)


# ----------------------------------------------------------
# MAIN RENDER FUNCTION
# ----------------------------------------------------------
def apply_design(room_image, texture_image, mask_image, zoom, brightness, contrast, saturation):
    """
    Combines room image, tiled wallpaper, and binary mask into final visualization.
    Handles resizing and normalization automatically.
    """
    if room_image is None or texture_image is None or mask_image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    room_h, room_w = room_image.shape[:2]
    tiled_texture = tile_texture(texture_image, room_w, room_h, zoom)

    lit_design = apply_lighting_and_color(room_image, tiled_texture)
    lit_design = apply_brightness_contrast(lit_design, brightness, contrast)
    lit_design = apply_saturation(lit_design, saturation)

    # --- Normalize mask to binary float 0..1 ---
    if len(mask_image.shape) == 3:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_image.copy()

    _, mask_gray = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY)
    mask_norm = (mask_gray.astype(np.float32) / 255.0).clip(0, 1)
    mask_3ch = np.repeat(mask_norm[:, :, np.newaxis], 3, axis=2)

    # --- Resize everything to same shape ---
    rh, rw = room_image.shape[:2]
    if mask_3ch.shape[:2] != (rh, rw):
        mask_3ch = cv2.resize(mask_3ch, (rw, rh), interpolation=cv2.INTER_NEAREST)
    if lit_design.shape[:2] != (rh, rw):
        lit_design = cv2.resize(lit_design, (rw, rh), interpolation=cv2.INTER_LINEAR)

    # --- Blend wallpaper and room ---
    room_f = room_image.astype(np.float32)
    lit_f = lit_design.astype(np.float32)
    final = room_f * (1 - mask_3ch) + lit_f * mask_3ch

    return np.clip(final, 0, 255).astype(np.uint8)
