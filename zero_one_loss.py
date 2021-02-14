#!/usr/bin/env python
# coding=utf-8

def zero_one_loss(y, y_pred):
    if y == y_pred:
        return 0
    else:
        return 1

