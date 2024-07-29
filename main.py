import cv2
import numpy as np 


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "point: %.4f, %.4f\n" % (self.x, self.y)


def p2p_dist(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def p2l_dist(p1, line1):
    return np.sqrt(line1.infer(p1)) / np.sqrt(line1.a**2, line1.b**2)


class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.a, self.b, self.c = (p2.y - p1.y, p1.x - p2.x, p1.y * p2.x - p1.x * p2.y)
        s = min(abs(self.a), abs(self.b), abs(self.c)) + 1
        self.a /= s 
        self.b /= s 
        self.c /= s 

    def set_abc(self, a, b, c):
        s = min(abs(a), abs(b), abs(c)) + 1
        self.a = a / s
        self.b = b / s
        self.c = c / s
        if self.a == 0:
            self.p1 = Point(0, -c / b)
            self.p2 = Point(1, -c / b)
            return
        if self.b == 0:
            self.p1 = Point(-c / a, 0)
            self.p2 = Point(-c / a, 1)
            return
        self.p1 = Point(0, -c / b)
        self.p2 = Point(-c / a, 0)
        
        
    def infer(self, p):
        return self.a * p.x + self.b * p.y + self.c

    def check(self, p):
        return abs(self.infer(p)) < 1e-7

    def __str__(self):
        return "line: %.4f, %.4f, %.4f\n" % (self.a, self.b, self.c)


class Circle:
    def __init__(self, p1, p2, point3):
        self.center = p1
        self.radius = p2p_dist(p2, point3)

    def set_cr(self, x, y, r):
        self.center = Point(x, y)
        self.radius = r

    def infer(self, p):
        return (p.x - self.center.x)**2 + (p.y - self.center.y)**2 - self.radius**2

    def check(self, p):
        return abs(self.infer(p)) < 1e-7

    def __str__(self):
        return "Circle: %.4f, %.4f, %.4f\n" % (self.center.x, self.center.y, self.radius)
        

def line2line(line1 : Line, line2 : Line):
    t = line1.a * line2.b - line1.b * line2.a
    if abs(t) < 1e-7:
        return None
    
    x = (line2.c * line1.b - line2.b * line1.c) / t
    y = -(line2.c * line1.a - line2.a * line1.c) / t
    p = Point(x, y)
    assert line1.check(p) and line2.check(p)
    return p


def circle2line(circle1 : Circle, line1: Line):
    c = line1.infer(circle1.center)
    div = (line1.a**2 + line1.b**2)
    d2 = div * circle1.radius ** 2 - c**2
    if d2 < 0:
        return None, None
    if d2 > 0:
        d = np.sqrt(d2)
        x1 = (-line1.a * c + abs(line1.b) * d) / div + circle1.center.x
        x2 = (-line1.a * c - abs(line1.b) * d) / div + circle1.center.x
        y1 = (-line1.b * c + abs(line1.a) * d) / div + circle1.center.y
        y2 = (-line1.b * c - abs(line1.a) * d) / div + circle1.center.y 
        p = Point(x1, y1)
        if line1.check(p):
            q = Point(x2, y2)
            return p, q
        p = Point(x1, y2)
        q = Point(x2, y1)
        return p, q

    x = -line1.a * c / div + circle1.center.x 
    y = -line1.b * c / div + circle1.center.y
    p = Point(x, y)
    assert circle1.check(p) and line1.check(p)
    return p, None


def circle2circle(circle1: Circle, circle2: Circle):
    c = circle1.infer(Point(0, 0)) - circle2.infer(Point(0, 0))
    a = 2 * (circle2.center.x - circle1.center.x)
    b = 2 * (circle2.center.y - circle1.center.y)
    l = Line(Point(0,0), Point(0,1))
    l.set_abc(a, b, c)
    return circle2line(circle1, l)


class Canvas:
    def __init__(self):
        self.width = 1280
        self.height = 1920

        self.circles = []
        self.lines = []
        self.intersections = []
        self.mainWindow = cv2.namedWindow('main')
        cv2.setMouseCallback('main', self.click_and_loc)

        self.mouse_pos = []
        self.mode = 0 # 0 for line 1 for circle 2 for remove

    def find(self, pos):
        p = Point(*pos)
        min_dist = 1e7
        best_p = None
        for item in self.intersections:
            d = p2p_dist(item, p)
            if d < min_dist:
                min_dist = d 
                best_p = item
        
        if min_dist < 5:
            return best_p
        return p

    def click_and_loc(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.mouse_pos.append([x, y])

            if self.mode == 0 and len(self.mouse_pos) == 2:
                self.draw_line(self.find(self.mouse_pos[0]), self.find(self.mouse_pos[1]))
                self.mouse_pos = []
                self.render()
            if self.mode == 1 and len(self.mouse_pos) == 3:
                self.draw_circle(self.find(self.mouse_pos[0]), self.find(self.mouse_pos[1]), self.find(self.mouse_pos[2]))
                self.mouse_pos = []
                self.render()
            if self.mode == 2 and len(self.mouse_pos) == 1:
                q = self.find(self.mouse_pos[0])
                if q in self.intersections:
                    self.intersections.remove(q)
                self.mouse_pos = []
                self.render()
        
        if event == cv2.EVENT_RBUTTONUP:
            self.mode = (1 + self.mode) % 3
            self.render()

    def add_inter(self, point):
        if point is not None:
            self.intersections.append(point)

    def draw_line(self, p1, p2):
        self.lines.append(Line(p1, p2))
        self.add_inter(p1)
        self.add_inter(p2)

        for item in self.lines[:-1]:
            p = line2line(item, self.lines[-1])
            self.add_inter(p)
        
        for item in self.circles:
            p, q = circle2line(item, self.lines[-1])
            self.add_inter(p)
            self.add_inter(q)

    def draw_circle(self, p1, p2, point3):
        self.circles.append(Circle(p1, p2, point3))
        for item in self.lines:
            p, q = circle2line(self.circles[-1], item)
            self.add_inter(p)
            self.add_inter(q)
        for item in self.circles[:-1]:
            p, q = circle2circle(self.circles[-1], item)
            self.add_inter(p)
            self.add_inter(q)

    def start(self):
        while True:
            res = self.render()
            if res == ord('q'):
                break

    def render(self):
        img = np.ones((self.width, self.height, 3), dtype='uint8') * 255
        erase = np.zeros((self.width, self.height), dtype='uint8')

        for item in self.circles:
            cv2.circle(img, (int(item.center.x), int(item.center.y)), int(item.radius), (0,0,0), thickness=1)

        for item in self.intersections:
            cv2.circle(img, (int(item.x), int(item.y)), 2, (0,0,0), thickness=-1)
            cv2.circle(erase, (int(item.x), int(item.y)), 10, 1, thickness=-1)

        img[erase == 0] = np.array([255, 255, 255])

        for item in self.lines:
            cv2.line(img, (int(item.p1.x), int(item.p1.y)), (int(item.p2.x), int(item.p2.y)), (0,0,0), thickness=1)

        if self.mode == 0:
            cv2.putText(img, "LINE", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), thickness=1)
        if self.mode == 1:
            cv2.putText(img, "CIRCLE", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), thickness=1)
        if self.mode == 2:
            cv2.putText(img, "REMOVE", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), thickness=1)
        cv2.imshow('main', img)
        key = cv2.waitKey(30)
        return key


def main():
    a = Point(0, 1)
    b = Point(1, 2)
    c = Point(100, 0)
    d = Point(20, 80)

    l1 = Line(a, b)
    l2 = Line(c, d)
    q = line2line(l1, l2)

    c1 = Circle(a, a, c)
    c2 = Circle(b, b, c)
    print(c1)
    print(c2)
    q = circle2circle(c1, c2)
    print(q[0])
    print(q[1])

    canvas = Canvas()
    canvas.start()


if __name__ == '__main__':
    main()