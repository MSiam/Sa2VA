import numpy as np
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore


class DirectResize:
    def __init__(self, target_length: int, both_sides : bool = True) -> None:
        self.target_length = target_length
        self.both_sides = both_sides

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        if self.both_sides:
            return np.array(img.resize((self.target_length, self.target_length)))
        else:
            width, height = img.size
            if width < height:
                scale_factor = self.target_length / width
                new_width = self.target_length
                new_height = int(height * scale_factor)
            else:
                scale_factor = self.target_length / height
                new_height = self.target_length
                new_width = int(width * scale_factor)

            return np.array(img.resize((new_width, new_height)))
