import cv2

class VideoPlay:
    def __init__(self, src):
        self.src = cv2.VideoCapture(src)
        self.window = cv2.namedWindow('x')

    @property
    def shape(self):
        if self.src.isOpened():
            ret, frame = self.src.read()
            return frame.shape
        else:
            return None

    def play(self):
        H, W = self.shape[:2]
        st = W//4
        ed = W//4 * 3
        cen = W//2
        while self.src.isOpened():
            ret, frame = self.src.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = frame[:,::-1]
                frame = (frame//16*16)
                cv2.imshow('x', frame)
                key = cv2.waitKey(30)
                if key == ord('q'):
                    return 1
        return -1

if __name__ == '__main__':
    play = VideoPlay(0)
    L = play.play()
