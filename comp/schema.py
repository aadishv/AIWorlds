import json
import random


class Detection:
    """
    Represents a detected object with bounding box (x, y, width, height),
    class id (cls), and depth.
    """

    VALID_CLASSES = {0, 1, 2, 3}

    def __init__(
        self, x: float, y: float, width: float, height: float, cls: int, depth: float
    ):
        # Validate coordinates and sizes
        for name, value in (("x", x), ("y", y), ("width", width), ("height", height)):
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number, got {type(value).__name__}")
            if not (0.0 <= value <= 640.0):
                raise ValueError(f"{name} must be between 0 and 640, got {value}")

        # Validate class id
        if not isinstance(cls, int):
            raise TypeError(f"cls must be an integer, got {type(cls).__name__}")
        if cls not in self.VALID_CLASSES:
            raise ValueError(
                f"cls must be one of {sorted(self.VALID_CLASSES)}, got {cls}"
            )

        # Validate depth
        if not isinstance(depth, (int, float)):
            raise TypeError(f"depth must be a number, got {type(depth).__name__}")

        # Assign
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)
        self.cls = cls
        self.depth = float(depth)

    def __repr__(self):
        return (
            f"Detection(x={self.x}, y={self.y}, width={self.width}, "
            f"height={self.height}, cls={self.cls}, depth={self.depth})"
        )

    def serialize(self) -> str:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "cls": self.cls,
            "depth": self.depth,
        }

    @staticmethod
    def random():
        """
        Generate a random Detection instance.
        Coordinates and sizes are uniform in [0, 640].
        cls is chosen from VALID_CLASSES.
        depth is a uniform float in [0, 1000].
        """
        x = random.uniform(0.0, 640.0)
        y = random.uniform(0.0, 640.0)
        width = random.uniform(0.0, 640.0)
        height = random.uniform(0.0, 640.0)
        cls = random.choice(list(Detection.VALID_CLASSES))
        depth = random.uniform(0.0, 1000.0)
        return Detection(x, y, width, height, cls, depth)
