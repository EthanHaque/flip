import cv2

from flip.transformers.element import Element
from flip.transformers.transformer import Transformer


class Resize(Transformer):
    """ Resize image of Element

    Parameters
    ----------
    mode: {'fixed', 'max_width', 'max_height'}, default='fixed'
        The mode of resizing.
    width: int, required if mode is 'fixed' or 'max_width'
        The width to resize the image to.
    height: int, required if mode is 'fixed' or 'max_height'
        The height to resize the image to.
    """

    def __init__(self, mode='fixed', width=None, height=None):
        self.mode = mode
        self.width = width
        self.height = height
        self._validate_parameters()


    def _validate_parameters(self):
        if self.mode == 'fixed':
            if self.width is None or self.height is None:
                raise ValueError("Both width and height are required for 'fixed' mode.")
        elif self.mode == 'max_width':
            if self.width is None:
                raise ValueError("Width is required for 'max_width' mode.")
        elif self.mode == 'max_height':
            if self.height is None:
                raise ValueError("Height is required for 'max_height' mode.")
        else:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Choose from 'fixed', 'max_width', or 'max_height'.")


    def map(self, element: Element, parent=None) -> Element:
        assert element, "Element cannot be None"
        original_height, original_width = element.image.shape[:2]

        if self.mode == 'fixed':
            element.image = cv2.resize(element.image, (self.width, self.height))
        elif self.mode == 'max_width':
            aspect_ratio = original_height / original_width
            new_width = self.width
            new_height = int(new_width * aspect_ratio)
            element.image = cv2.resize(element.image, (new_width, new_height))
        elif self.mode == 'max_height':
            aspect_ratio = original_width / original_height
            new_height = self.height
            new_width = int(new_height * aspect_ratio)
            element.image = cv2.resize(element.image, (new_width, new_height))

        return element
