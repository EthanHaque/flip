import numpy as np

from flip.transformers.element import Element
from flip.transformers.transformer import Transformer
from flip.utils import rotate_bound, crop_from_angle

class Rotate(Transformer):
    """ Flip image of Element

        Parameters
        ----------
        mode : {'random', 'by_angle', '90', 'upside_down'}, default='random'
    """

    _SUPPORTED_MODES = {'random', '90', 'upside_down', 'by_angle'}

    def __init__(self, mode='random', angle=0, min=0, max=360, force=True, crop=True):
        self.mode = mode
        self.force = force
        self.crop = crop

        if self.mode not in self._SUPPORTED_MODES:
            raise ValueError("Mode '{0:s}' not supported. ".format(self.mode))

        if self.mode == 'upside_down':
            self.angles = [0, 180]
        elif self.mode == '90':
            self.angles = [0, 90, 180, 270]
        elif self.mode == 'by_angle':
            self.angles = [angle]
        else:
            self.angles = [min, max]

    def map(self, element: Element, parent=None) -> Element:
        assert element, "Element cannot be None"
        if self.mode == 'upside_down' or self.mode == '90':
            angle = np.random.choice(self.angles)
        elif self.mode == 'by_angle':
            angle = self.angles[0]
        else:
            angle = np.random.uniform(low=self.angles[0], high=self.angles[1],)

        old_width = element.image.shape[1]
        old_height = element.image.shape[0]

        if self.force == False:
            if np.random.randint(low=0, high=2) == 0:
                element.image = rotate_bound(element.image, angle)
                element.angle = angle
                if self.crop:
                    element.image = crop_from_angle(element.image, old_width, old_height, -angle)
            else:
                element.angle = 0
        else:
            element.image = rotate_bound(element.image, angle)
            element.angle = angle
            if self.crop:
                element.image = crop_from_angle(element.image, old_width, old_height, -angle)
        return element
