#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np
from numpy import random as rd

import zero_one_loss as zo
import hinge_loss as hl


def main():
    t = tf.constant(0.)
    y = tf.linspace(-1., 1., 500)
    y_pred = np.rd.rand(10)

    zo.zero_one_loss(y, y_pred)
    
    tf.Session.run(hl.hinge_loss(t, y))

if __name__ == '__main__':
    main()
