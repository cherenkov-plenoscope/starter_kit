import numpy as np
from . import Vec2

class Ray2:
    def __init__(self, support, direction):
        self.support = support
        self.direction = Vec2.normalize(direction)


def at(ray, alpha):
    return Vec2.add(ray.support, Vec2.multiply(ray.direction, alpha))
