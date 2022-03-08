import numpy as np

class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def add(a, b):
    return Vec2(x=(a.x + b.x), y=(a.y + b.y))


def subtract(a, b):
    return Vec2(x=(a.x - b.x), y=(a.y - b.y))


def multiply(v, scalar):
    return Vec2(x=(v.x*scalar), y=(v.y*scalar))


def norm(a):
    return np.hypot(a.x, a.y)


def normalize(a):
    return multiply(v=a, scalar=1.0/(norm(a=a)))
