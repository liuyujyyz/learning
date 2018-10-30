import numpy as np
import cv2

class Point:
    def __init__(self, x, y, z):
        self.position = (x, y, z)

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]

    def __sub__(self, point):
        return (self.x - point.x, self.y - point.y, self.z - point.z)


def intermedia(pa, pb, delta):
    x = pa.x * delta + pb.x * (1-delta)
    y = pa.y * delta + pb.y * (1-delta)
    z = pa.z * delta + pb.z * (1-delta)
    return Point(x, y, z)


class Line:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    @property
    def length(self):
        v = self.start - self.end
        re = 0
        for i in v:
            re += i**2
        re = re**0.5
        return re

    def intersect(self, line):
        pass

if __name__ == '__main__':
    p = Point(1,2,3)
    q = Point(4,5,6)
    r = Point(7,8,9)
    l1 = Line(p, q)
    l2 = Line(q, r)
    print(l1.length, l2.length)


