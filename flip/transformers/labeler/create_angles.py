from flip.transformers.element import Element
from flip.transformers.transformer import Transformer


class CreateAngles(Transformer):
    def map(self, element: Element) -> Element:
        assert element, "element cannot be None"

        if element.tags is None:
            element.tags = self.create(element)
        else:
            element.tags = element.tags + self.create(element)

        return element
    
    def create(self, element):
        array = []
        for obj in element.objects:
            data = {"name": obj.name, "angle": obj.angle}

            array.append(data)

        return array
    

    