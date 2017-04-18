import tensorflow as tf
import cv2
import time

class MonitorWriter(object):
    def __init__(self, filename):
        if filename is None:
            filename = time.time()
        self.writer = tf.summary.FileWriter('../data/' + str(filename))

    def add_value(self, tag, value, step):
        self.writer.add_summary(tf.summary.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)]), global_step = step)

    def add_image(self, tag, value, step=None):
        if isinstance(value, str):
            self.writer.add_summary(tf.summary.Summary(value=[tf.Summary.Value(tag=tag, image=tf.Summary.Image(encoded_image_string = value))]), global_step=step)
        else:
            if value.max() < 2:
                value = (value * 256).astype('uint8')
            cv2.imwrite('/tmp/a.jpg', value)
            self.writer.add_summary(tf.summary.Summary(value=[tf.Summary.Value(tag=tag, image=tf.Summary.Image(encoded_image_string = open('/tmp/a.jpg','rb').read()))]), global_step=step)


