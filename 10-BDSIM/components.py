import numpy

class HalbachPMQ:
    def __init__(self, aper, b_r=1.2):
        self.aper = aper
        self.b_r = b_r
        self.grad = (2 * self.b_r) / self.aper
