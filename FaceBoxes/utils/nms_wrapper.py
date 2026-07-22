# coding: utf-8

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

try:
    from .nms.cpu_nms import cpu_nms, cpu_soft_nms
except ImportError:
    from .nms.py_cpu_nms import py_cpu_nms

    cpu_nms = py_cpu_nms
    cpu_soft_nms = None


def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    return cpu_nms(dets, thresh)
    # return gpu_nms(dets, thresh)
