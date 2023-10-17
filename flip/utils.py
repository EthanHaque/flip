import cv2
import numpy as np
import largestinteriorrectangle as lir


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


def compute_rotated_bounding_box(width, height, angle):
    theta_rad = np.radians(angle)
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float64)
    cx, cy = width / 2, height / 2
    corners -= np.array([cx, cy])
    
    rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                                [np.sin(theta_rad), np.cos(theta_rad)]])
    rotated_corners = np.dot(corners, rotation_matrix.T) + np.array([cx, cy])
    rotated_corners -= rotated_corners.min(axis=0)
    rotated_corners = np.array([rotated_corners[[1, 0, 2, 3]]], dtype=np.int32)

    return rotated_corners


def crop_from_angle(rotated_image, old_width, old_height, angle):
    box = compute_rotated_bounding_box(old_width, old_height, angle)
    rectangle = lir.lir(box)
    x, y, w, h = rectangle

    x1 = x + 1
    y1 = y + 1
    x2 = x + w - 1
    y2 = y + h - 1

    return rotated_image[y1:y2, x1:x2]
