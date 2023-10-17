import cv2
import numpy as np
import math


def inv_channels(image):
    image[..., :3] = image[..., (2, 1, 0)]
    return image


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    scale = 1.0

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, scale)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(
        image,
        M,
        (nW, nH)
    )


def overlay_transparent(
    background: np.ndarray,
    overlay: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
    hist_match=False,
    min_object_brightness=0,
):

    background_width = background.shape[1]
    background_height = background.shape[0]

    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    x = x
    y = y
    if x==None or y==None:
        return background
    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]
        mask = mask[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]
        mask = mask[:h]

    mask = mask / 255.0

    if mask.ndim < background.ndim:
        mask = np.expand_dims(mask, axis=-1)

    if overlay.ndim < background.ndim:
        overlay = np.expand_dims(overlay, axis=-1)

    if background.ndim > 2 and background.shape[2] > 3:
        background = background[..., :3]

    if hist_match:
        overlay = histogram_matching_h(
            overlay, background[y : y + h, x : x + w], min_v=min_object_brightness
        )
    background[y : y + h, x : x + w] = (1.0 - mask) * background[
        y : y + h, x : x + w
    ] + mask * overlay
    return background


def crop_from_contour(image):
    if image.shape[2] != 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
    _, _, _, a = cv2.split(image)
    ret, thresh = cv2.threshold(a, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    index = 0
    for i in range(len(contours)):
        if len(contours[i]) > len(contours[index]):
            index = i
    rect = cv2.boundingRect(contours[index])
    cropped_img = image[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]

    return cropped_img


def largest_inscribed_rectangle(width, height, angle):
    if width <= 0 or height <= 0:
        return 0, 0

    width_is_longer = width >= height
    longer_side, shorter_side = (width, height) if width_is_longer else (height, width)

    abs_sin_angle = abs(math.sin(angle))
    abs_cos_angle = abs(math.cos(angle))

    # Determine if the rectangle is half or fully constrained, and calculate dimensions
    if shorter_side <= 2 * abs_sin_angle * abs_cos_angle * longer_side or \
            abs(abs_sin_angle - abs_cos_angle) < 1e-10:
        # Half constrained case
        x = 0.5 * shorter_side
        inscribed_width, inscribed_height = (x / abs_sin_angle, x / abs_cos_angle) if width_is_longer \
            else (x / abs_cos_angle, x / abs_sin_angle)
    else:
        # Fully constrained case
        cos_2angle = abs_cos_angle**2 - abs_sin_angle**2
        inscribed_width = (width * abs_cos_angle - height * abs_sin_angle) / cos_2angle
        inscribed_height = (height * abs_cos_angle - width * abs_sin_angle) / cos_2angle

    return inscribed_width, inscribed_height


def crop_from_angle(rotated_image, old_width, old_height, angle):
    wr, hr = largest_inscribed_rectangle(old_width, old_height, math.radians(angle))

    # Now, calculate the position where this rectangle should be cropped from the center of the rotated image
    center_x = rotated_image.shape[1] // 2
    center_y = rotated_image.shape[0] // 2
    
    x1 = max(0, center_x - int(wr // 2))
    y1 = max(0, center_y - int(hr // 2))
    x2 = min(rotated_image.shape[1], center_x + int(wr // 2))
    y2 = min(rotated_image.shape[0], center_y + int(hr // 2))

    return rotated_image[y1:y2, x1:x2]


if __name__ == "__main__":
    width, height = 700, 1000
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    angle = 97
    rotated_image = rotate_bound(image, angle)

    cropped_image = crop_from_angle(rotated_image, width, height, angle)
    cv2.imwrite("cropped_image.png", cropped_image)