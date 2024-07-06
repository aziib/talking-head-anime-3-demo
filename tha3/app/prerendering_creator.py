import argparse
import logging
import os
import sys
from enum import Enum
from typing import Optional, Dict, List

sys.path.append(os.getcwd())

import PIL.Image
import numpy
import torch
import wx
from torch.nn import functional as F
from numba import jit
import time
import math

# import pdb

from tha3.poser.modes.load_poser import load_poser
from tha3.poser.poser import Poser, PoseParameterCategory, PoseParameterGroup
from tha3.util import extract_pytorch_image_from_filelike, rgba_to_numpy_image, grid_change_to_numpy_image, \
    rgb_to_numpy_image, torch_linear_to_srgb, resize_PIL_image, extract_PIL_image_from_filelike, extract_pytorch_image_from_PIL_image

from tha3.mocap.ifacialmocap_pose import create_default_ifacialmocap_pose

from tha3.mocap.ifacialmocap_constants import RIGHT_EYE_BONE_X, RIGHT_EYE_BONE_Y, RIGHT_EYE_BONE_Z, LEFT_EYE_BONE_X, LEFT_EYE_BONE_Y, LEFT_EYE_BONE_Z, \
    HEAD_BONE_QUAT, LEFT_EYE_BONE_QUAT, RIGHT_EYE_BONE_QUAT
from tha3.mocap.ifacialmocap_constants import MOUTH_SMILE_LEFT, MOUTH_SHRUG_UPPER, MOUTH_SMILE_RIGHT, \
    BROW_INNER_UP, BROW_OUTER_UP_RIGHT, BROW_OUTER_UP_LEFT, BROW_DOWN_LEFT, BROW_DOWN_RIGHT, EYE_WIDE_LEFT, \
    EYE_WIDE_RIGHT, EYE_BLINK_LEFT, EYE_BLINK_RIGHT, CHEEK_SQUINT_LEFT, CHEEK_SQUINT_RIGHT, EYE_LOOK_IN_LEFT, \
    EYE_LOOK_OUT_LEFT, EYE_LOOK_IN_RIGHT, EYE_LOOK_OUT_RIGHT, EYE_LOOK_UP_LEFT, EYE_LOOK_UP_RIGHT, EYE_LOOK_DOWN_RIGHT, \
    EYE_LOOK_DOWN_LEFT, HEAD_BONE_X, HEAD_BONE_Y, HEAD_BONE_Z, JAW_OPEN, MOUTH_FROWN_LEFT, MOUTH_FROWN_RIGHT, \
    MOUTH_LOWER_DOWN_LEFT, MOUTH_LOWER_DOWN_RIGHT, MOUTH_FUNNEL, MOUTH_PUCKER

import tha3.mocap.ifacialmocap_add as ifadd

from tha3.poser.modes.pose_parameters import get_pose_parameters

class EyebrowDownMode(Enum):
    TROUBLED = 1
    ANGRY = 2
    LOWERED = 3
    SERIOUS = 4

class WinkMode(Enum):
    NORMAL = 1
    RELAXED = 2

def rad_to_deg(rad):
    return rad * 180.0 / math.pi


def deg_to_rad(deg):
    return deg * math.pi / 180.0


def clamp(x, min_value, max_value):
    return max(min_value, min(max_value, x))

def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    image = image.to(device)
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)

lower_smile_threshold: float = 0.4
upper_smile_threshold: float = 0.6
eyebrow_down_mode: EyebrowDownMode = EyebrowDownMode.ANGRY
wink_mode: WinkMode = WinkMode.NORMAL
eye_surprised_max_value: float = 0.5
eye_wink_max_value: float = 0.8
eyebrow_down_max_value: float = 0.4
cheek_squint_min_value: float = 0.1
cheek_squint_max_value: float = 0.7
eye_rotation_factor: float = 1.0 / 0.75
jaw_open_min_value: float = 0.1
jaw_open_max_value: float = 0.4
mouth_frown_max_value: float = 0.6
mouth_funnel_min_value: float = 0.25
mouth_funnel_max_value: float = 0.5
iris_small_left_default=0.0
iris_small_right_default=0.0
eye_blink_max_value = eye_wink_max_value
eye_wide_max_value = eye_surprised_max_value

pose_parameters = get_pose_parameters()
pose_size = 45

eyebrow_troubled_left_index = pose_parameters.get_parameter_index("eyebrow_troubled_left")
eyebrow_troubled_right_index = pose_parameters.get_parameter_index("eyebrow_troubled_right")
eyebrow_angry_left_index = pose_parameters.get_parameter_index("eyebrow_angry_left")
eyebrow_angry_right_index = pose_parameters.get_parameter_index("eyebrow_angry_right")
eyebrow_happy_left_index = pose_parameters.get_parameter_index("eyebrow_happy_left")
eyebrow_happy_right_index = pose_parameters.get_parameter_index("eyebrow_happy_right")
eyebrow_raised_left_index = pose_parameters.get_parameter_index("eyebrow_raised_left")
eyebrow_raised_right_index = pose_parameters.get_parameter_index("eyebrow_raised_right")
eyebrow_lowered_left_index = pose_parameters.get_parameter_index("eyebrow_lowered_left")
eyebrow_lowered_right_index = pose_parameters.get_parameter_index("eyebrow_lowered_right")
eyebrow_serious_left_index = pose_parameters.get_parameter_index("eyebrow_serious_left")
eyebrow_serious_right_index = pose_parameters.get_parameter_index("eyebrow_serious_right")

eye_surprised_left_index = pose_parameters.get_parameter_index("eye_surprised_left")
eye_surprised_right_index = pose_parameters.get_parameter_index("eye_surprised_right")
eye_wink_left_index = pose_parameters.get_parameter_index("eye_wink_left")
eye_wink_right_index = pose_parameters.get_parameter_index("eye_wink_right")
eye_happy_wink_left_index = pose_parameters.get_parameter_index("eye_happy_wink_left")
eye_happy_wink_right_index = pose_parameters.get_parameter_index("eye_happy_wink_right")
eye_relaxed_left_index = pose_parameters.get_parameter_index("eye_relaxed_left")
eye_relaxed_right_index = pose_parameters.get_parameter_index("eye_relaxed_right")
eye_raised_lower_eyelid_left_index = pose_parameters.get_parameter_index("eye_raised_lower_eyelid_left")
eye_raised_lower_eyelid_right_index = pose_parameters.get_parameter_index("eye_raised_lower_eyelid_right")

iris_small_left_index = pose_parameters.get_parameter_index("iris_small_left")
iris_small_right_index = pose_parameters.get_parameter_index("iris_small_right")

iris_rotation_x_index = pose_parameters.get_parameter_index("iris_rotation_x")
iris_rotation_y_index = pose_parameters.get_parameter_index("iris_rotation_y")

head_x_index = pose_parameters.get_parameter_index("head_x")
head_y_index = pose_parameters.get_parameter_index("head_y")
neck_z_index = pose_parameters.get_parameter_index("neck_z")

mouth_aaa_index = pose_parameters.get_parameter_index("mouth_aaa")
mouth_iii_index = pose_parameters.get_parameter_index("mouth_iii")
mouth_uuu_index = pose_parameters.get_parameter_index("mouth_uuu")
mouth_eee_index = pose_parameters.get_parameter_index("mouth_eee")
mouth_ooo_index = pose_parameters.get_parameter_index("mouth_ooo")

mouth_lowered_corner_left_index = pose_parameters.get_parameter_index("mouth_lowered_corner_left")
mouth_lowered_corner_right_index = pose_parameters.get_parameter_index("mouth_lowered_corner_right")
mouth_raised_corner_left_index = pose_parameters.get_parameter_index("mouth_raised_corner_left")
mouth_raised_corner_right_index = pose_parameters.get_parameter_index("mouth_raised_corner_right")

body_y_index = pose_parameters.get_parameter_index("body_y")
body_z_index = pose_parameters.get_parameter_index("body_z")
breathing_index = pose_parameters.get_parameter_index("breathing")

breathing_start_time = time.time()

expression_str = {
    0 : "base-aaa",
    1 : "base-iii",
    2 : "base-uuu",
    3 : "base-eee",
    4 : "base-ooo",
    5 : "base-raised",
    6 : "base-lowered",
    7 : "eye-normal-XY",
    8 : "eye-surprised-XY",
    9 : "eyelid-normal-happy",
    10 : "eyelid-relaxed",
    11 : "eyelid-lower",
    12 : "eyebrow-mask",
    13 : "eyebrow-troubled",
    14 : "eyebrow-angry",
    15 : "eyebrow-lowered",
    16 : "eyebrow-serious",
    17 : "eyebrow-happy",
    18 : "eyelid-surprised",
    19 : "flowmap-x",
    20 : "flowmap-head-y",
    21 : "flowmap-body-y",
    22 : "flowmap-z",
    23 : "flowmap-breath",
    24 : "fullimage",
    25 : "finished",
}

class PrerenderingMode(Enum):
    BaseMouth = 1
    EyeRotation = 2
    Eyelid = 3
    EyebrowMask = 4
    Eyebrow = 5
    RotationMap = 6
    SourceImage = 7
    Finished = -1

prerendering_count = [
    [10, 0, PrerenderingMode.BaseMouth], # 0
    [10, 0, PrerenderingMode.BaseMouth], # 1
    [10, 0, PrerenderingMode.BaseMouth], # 2
    [10, 0, PrerenderingMode.BaseMouth], # 3
    [10, 0, PrerenderingMode.BaseMouth], # 4
    [10, 0, PrerenderingMode.BaseMouth], # 5
    [10, 0, PrerenderingMode.BaseMouth], # 6
    [10, 10, PrerenderingMode.EyeRotation], # 7
    [10, 10, PrerenderingMode.EyeRotation], # 8
    [10, 10, PrerenderingMode.Eyelid], # 9
    [10, 0, PrerenderingMode.Eyelid], # 10
    [10, 0, PrerenderingMode.Eyelid], # 11
    [0, 0, PrerenderingMode.EyebrowMask], # 12
    [10, 0, PrerenderingMode.Eyebrow], # 13
    [10, 0, PrerenderingMode.Eyebrow], # 14
    [10, 0, PrerenderingMode.Eyebrow], # 15
    [10, 0, PrerenderingMode.Eyebrow], # 16
    [10, 0, PrerenderingMode.Eyebrow], # 17
    [0, 0, PrerenderingMode.Eyelid], # 18
    [1, 0, PrerenderingMode.RotationMap], # 19
    [1, 0, PrerenderingMode.RotationMap], # 20
    [1, 0, PrerenderingMode.RotationMap], # 21
    [1, 10, PrerenderingMode.RotationMap], # 22
    [1, 0, PrerenderingMode.RotationMap], # 23
    [0, 0, PrerenderingMode.SourceImage], # 24
    [0, 0, PrerenderingMode.Finished], # 25
]

def convert_output_image_from_torch_to_numpy(output_image):
    if output_image.shape[2] == 2:
        h, w, c = output_image.shape
        numpy_image = torch.transpose(output_image.reshape(h * w, c), 0, 1).reshape(c, h, w)
    elif output_image.shape[0] == 4:
        numpy_image = rgba_to_numpy_image(output_image)
    elif output_image.shape[0] == 3:
        numpy_image = rgb_to_numpy_image(output_image)
    elif output_image.shape[0] == 1:
        c, h, w = output_image.shape
        alpha_image = torch.cat([output_image.repeat(3, 1, 1) * 2.0 - 1.0, torch.ones(1, h, w)], dim=0)
        numpy_image = rgba_to_numpy_image(alpha_image)
    elif output_image.shape[0] == 2:
        numpy_image = grid_change_to_numpy_image(output_image, num_channels=4)
    else:
        raise RuntimeError("Unsupported # image channels: %d" % output_image.shape[0])
    numpy_image = numpy.uint8(numpy.rint(numpy_image * 255.0))
    return numpy_image

def map_preview(np_map, image_size = 512):
    np_map_preview = numpy.zeros((image_size, image_size, 4), dtype = numpy.uint8)

    np_map_preview[:, :, 0:2] = np_map + 128.0
    np_map_preview[:, :, 3] = 255

    return np_map_preview

@jit(nopython=True)
def map_np_sub(np_image_map, image_size = 512):
    np_output = numpy.zeros((image_size, image_size, 2), dtype = numpy.float32)
    np_flag = numpy.zeros((image_size, image_size, 1), dtype = numpy.uint8)
    for y in range(image_size):
        for x in range(image_size):
            target_x = min(image_size - 1, max(0, x + round(np_image_map[0, y, x])))
            target_y = min(image_size - 1, max(0, y + round(np_image_map[1, y, x])))
            np_output[target_y, target_x, 0] = np_image_map[1, y, x]
            np_output[target_y, target_x, 1] = np_image_map[0, y, x]
            np_flag[target_y, target_x, 0] = 1

    for x in range(0, image_size):
        if np_flag[0, x, 0] == 0:
            np_output[0, x] = [0.0, 0.0]
            np_flag[0, x, 0] = 1
    for y in range(1, image_size):
        if np_flag[y, 0, 0] == 0:
            np_output[y, 0] = [0.0, 0.0]
            np_flag[y, 0, 0] = 1
        if np_flag[y, image_size - 1, 0] == 0:
            np_output[y, image_size - 1] = [0.0, 0.0]
            np_flag[y, image_size - 1, 0] = 1

    for y in range(1, image_size - 1):
        for x in range(1, image_size - 1):
            if np_flag[y, x, 0] == 0:
                np_temp = np_output[y-1, x-1] + np_output[y-1, x] + np_output[y-1, x+1] + \
                          np_output[y, x-1] + np_output[y, x+1] + \
                          np_output[y+1, x-1] + np_output[y+1, x] + np_output[y+1, x+1]
                flag_count = np_flag[y-1, x-1] + np_flag[y-1, x] + np_flag[y-1, x+1] + \
                          np_flag[y, x-1] + np_flag[y, x+1] + \
                          np_flag[y+1, x-1] + np_flag[y+1, x] + np_flag[y+1, x+1]
                np_output[y, x] = np_temp / flag_count[0]
                np_flag[y, x, 0] = 1

    for x in range(1, image_size - 1):
        if np_flag[image_size - 1, x, 0] == 0:
                np_temp = np_output[image_size - 2, x-1] + np_output[image_size - 2, x] + np_output[image_size - 2, x+1] + \
                          np_output[image_size - 1, x-1] + np_output[image_size - 1, x+1]
                flag_count = np_flag[image_size - 2, x-1] + np_flag[image_size - 2, x] + np_flag[image_size - 2, x+1] + \
                          np_flag[image_size - 1, x-1] + np_flag[image_size - 1, x+1]
                np_output[image_size - 1, x] = np_temp / flag_count[0]
                np_flag[image_size - 1, x, 0] = 1
    return np_output


def map_np_convert(torch_map, image_size = 512):
    torch_image_map = torch_map * 256.0

    np_image_map = torch_image_map.detach().cpu().numpy()

    np_output = map_np_sub(np_image_map, image_size)

    return np_output

def map_np_convert2(torch_map, image_size = 512):
    torch_image_map = torch_map * 256.0

    np_image_map = torch_image_map.detach().cpu().numpy()

    np_output = numpy.zeros((image_size, image_size, 2), dtype = numpy.float32)
    np_flag = numpy.zeros((image_size, image_size, 1), dtype = numpy.uint8)
    for y in range(image_size):
        for x in range(image_size):
            target_x = min(image_size - 1, max(0, x + round(np_image_map[0, y, x])))
            target_y = min(image_size - 1, max(0, y + round(np_image_map[1, y, x])))
            np_output[target_y, target_x, 0] = np_image_map[1, y, x]
            np_output[target_y, target_x, 1] = np_image_map[0, y, x]
            np_flag[target_y, target_x, 0] = 1

    for x in range(0, image_size):
        if np_flag[0, x, 0] == 0:
            np_output[0, x] = [0.0, 0.0]
            np_flag[0, x, 0] = 1
    for y in range(1, image_size):
        if np_flag[y, 0, 0] == 0:
            np_output[y, 0] = [0.0, 0.0]
            np_flag[y, 0, 0] = 1
        if np_flag[y, image_size - 1, 0] == 0:
            np_output[y, image_size - 1] = [0.0, 0.0]
            np_flag[y, image_size - 1, 0] = 1

    for y in range(1, image_size - 1):
        for x in range(1, image_size - 1):
            if np_flag[y, x, 0] == 0:
                np_temp = np_output[y-1:y+2, x-1:x+2].sum(axis=(0, 1))
                flag_count = np_flag[y-1:y+2, x-1:x+2].sum(axis=(0, 1))
                np_output[y, x] = np_temp / flag_count[0]
                np_flag[y, x, 0] = 1

    for x in range(1, image_size - 1):
        if np_flag[image_size - 1, x, 0] == 0:
                np_temp = np_output[image_size - 2:image_size, x-1:x+2].sum(axis=(0, 1))
                flag_count = np_flag[image_size - 2:image_size, x-1:x+2].sum(axis=(0, 1))
                np_output[image_size - 1, x] = np_temp / flag_count[0]
                np_flag[image_size - 1, x, 0] = 1
    return np_output

def map_image_convert(torch_map, image_size = 512):
    torch_image_map = torch_map * 256.0

    np_image_map_temp = torch_image_map.int()
    np_image_map = np_image_map_temp.detach().cpu().numpy()

    np_output = numpy.zeros((image_size, image_size, 4), dtype = numpy.int64)
    for y in range(image_size):
        for x in range(image_size):
            target_x = min(image_size - 1, max(0, x + np_image_map[1, x, y]))
            target_y = min(image_size - 1, max(0, y + np_image_map[0, x, y]))
            np_output[target_x, target_y, 0] = np_image_map[1, x, y] + 128
            np_output[target_x, target_y, 1] = np_image_map[0, x, y] + 128
            np_output[target_x, target_y, 3] = 1

    for x in range(0, image_size):
        if np_output[x, 0, 3] == 0:
            np_output[x, 0] = [128, 128, 0, 1]
    for y in range(1, image_size):
        if np_output[0, y, 3] == 0:
            np_output[0, y] = [128, 128, 0, 1]
        if np_output[image_size - 1, y, 3] == 0:
            np_output[image_size - 1, y] = [128, 128, 0, 1]

    for y in range(1, image_size - 1):
        for x in range(1, image_size - 1):
            if np_output[x, y, 3] == 0:
                np_output[x, y] = np_output[x-1:x+2, y-1:y+2].sum(axis=(0, 1))
                np_output[x, y] = np_output[x, y] / np_output[x, y, 3]

    for x in range(1, image_size - 1):
        if np_output[x, y, 3] == 0:
            np_output[x, y] = np_output[x-1:x+2, y-1:y+1].sum(axis=(0, 1))
            np_output[x, y] = np_output[x, y] / np_output[x, y, 3]

    for y in range(image_size):
        for x in range(image_size):
            np_output[x, y, 3] = 255

    np_output = np_output.clip(0, 255).astype('uint8')
    return np_output

@jit(nopython=True)
def numpy_map_convert(np_map, np_address, image_size = 512):
    pass
    if True:
        np_output = numpy.zeros((image_size, image_size, 4), dtype = numpy.int64)
        for x in range(image_size):
            for y in range(image_size):
                np_output[np_address[1, x, y], np_address[0, x, y]][0] = np_map[1][x, y]
                np_output[np_address[1, x, y], np_address[0, x, y]][1] = np_map[0][x, y]
                np_output[np_address[1, x, y], np_address[0, x, y]][3] = 255
        return np_output

@jit(nopython=True)
def numpy_map_convert2(np_map, image_size = 512):
    pass
    if True:
        np_output = numpy.zeros((image_size, image_size, 4), dtype = numpy.int64)
        for x in range(image_size):
            for y in range(image_size):
                target_x = min(image_size, max(0, x + np_map[1, x, y]))
                target_y = min(image_size, max(0, y + np_map[0, x, y]))
                np_output[target_x, target_y] = [np_map[1, x, y] + 128, np_map[0, x, y] + 128, 0, 255]
        np_output = np_output.clip(0, 255).astype('uint8')
        return np_output

class MainFrame(wx.Frame):
    def __init__(self, poser: Poser, device: torch.device):
        super().__init__(None, wx.ID_ANY, "THA3 Prerendering Creator")
        self.poser = poser
        self.dtype = self.poser.get_dtype()
        self.device = device
        self.image_size = self.poser.get_image_size()

        self.wx_source_image = None
        self.torch_source_image = None

        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(self.main_sizer)
        self.SetAutoLayout(1)
        self.init_left_panel()
        self.init_right_panel()
        self.main_sizer.Fit(self)

        self.last_pose = None
        self.last_output_numpy_image = None

        self.wx_source_image = None
        self.torch_source_image = None
        self.source_image_bitmap = wx.Bitmap(self.image_size, self.image_size)
        self.result_image_bitmap = wx.Bitmap(self.image_size, self.image_size)
        self.source_image_dirty = True

        self.source_image_string = None

        self.image_filename_base = None
        self.image_filepath_base = None
        self.image_savepath = None

        self.mask_torch_image = None

        self.expression_index = 0
        self.index_param_0 = 0
        self.index_param_1 = 0

        self.mouth_open_ratio = 1.0

        self.create_timers()
        self.Bind(wx.EVT_CLOSE, self.on_close)


    def init_left_panel(self):
        self.left_panel = wx.Panel(self, style=wx.SIMPLE_BORDER)
        left_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.left_panel.SetSizer(left_panel_sizer)
        self.left_panel.SetAutoLayout(1)

        self.source_image_panel = wx.Panel(self.left_panel, size=(self.image_size, self.image_size),
                                           style=wx.SIMPLE_BORDER)
        self.source_image_panel.Bind(wx.EVT_PAINT, self.paint_source_image_panel)
        self.source_image_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
        left_panel_sizer.Add(self.source_image_panel, 0, wx.FIXED_MINSIZE)

        self.load_image_button = wx.Button(self.left_panel, wx.ID_ANY, "\nLoad Image\n\n")
        left_panel_sizer.Add(self.load_image_button, 1, wx.EXPAND)
        self.load_image_button.Bind(wx.EVT_BUTTON, self.load_image)

        left_panel_sizer.Fit(self.left_panel)
        self.main_sizer.Add(self.left_panel, 0, wx.FIXED_MINSIZE)

    def on_erase_background(self, event: wx.Event):
        pass

    def init_right_panel(self):
        self.right_panel = wx.Panel(self, style=wx.SIMPLE_BORDER)
        right_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_panel.SetSizer(right_panel_sizer)
        self.right_panel.SetAutoLayout(1)

        self.result_image_panel = wx.Panel(self.right_panel,
                                           size=(self.image_size, self.image_size),
                                           style=wx.SIMPLE_BORDER)
        self.result_image_panel.Bind(wx.EVT_PAINT, self.paint_result_image_panel)
        self.result_image_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
        right_panel_sizer.Add(self.result_image_panel, 0, wx.FIXED_MINSIZE)

        self.msg_text = wx.StaticText(self.right_panel, label="")
        right_panel_sizer.Add(self.msg_text, wx.SizerFlags().Border())

        right_panel_sizer.Fit(self.right_panel)
        self.main_sizer.Add(self.right_panel, 0, wx.FIXED_MINSIZE)

    def load_image(self, event: wx.Event):
        dir_name = "data/images"
        file_dialog = wx.FileDialog(self, "Choose an image", dir_name, "", "*.png", wx.FD_OPEN)
        if file_dialog.ShowModal() == wx.ID_OK:
            image_file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
            try:
                pil_image = resize_PIL_image(extract_PIL_image_from_filelike(image_file_name),
                                             (self.poser.get_image_size(), self.poser.get_image_size()))
                w, h = pil_image.size
                if pil_image.mode != 'RGBA':
                    self.source_image_string = "Image must have alpha channel!"
                    self.wx_source_image = None
                    self.torch_source_image = None
                else:
                    self.image_filepath_base = file_dialog.GetDirectory()
                    self.image_filename_base = os.path.splitext(file_dialog.GetFilename())[0]
                    self.image_savepath = os.path.join(self.image_filepath_base, self.image_filename_base + "_tha")
                    if os.path.isdir(self.image_savepath) == False:
                        os.makedirs(self.image_savepath)
                    self.expression_index = 0
                    self.index_param_0 = 0
                    self.index_param_1 = 0
                    self.mask_torch_image = None
                    self.source_image_string = None
                    self.wx_source_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                    self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image)\
                        .to(self.device).to(self.dtype)

                    # Detection for Mouth Open in Image
                    maskpose = [0.0 for i in range(pose_size)]
                    maskpose[mouth_aaa_index] = 1.0
                    with torch.no_grad():
                        maskpose_tensor = torch.tensor(maskpose, device=self.device, dtype=self.dtype)
                        mask_mouth_open = self.poser.pose(self.torch_source_image, maskpose_tensor, 12)[0].detach().cpu()
                    maskpose = [0.0 for i in range(pose_size)]
                    maskpose[mouth_aaa_index] = 0.0
                    with torch.no_grad():
                        maskpose_tensor = torch.tensor(maskpose, device=self.device, dtype=self.dtype)
                        mask_mouth_close = self.poser.pose(self.torch_source_image, maskpose_tensor, 12)[0].detach().cpu()
                    mouth_open_val = torch.sum(mask_mouth_open > 0.5).item()
                    mouth_close_val = torch.sum(mask_mouth_close > 0.5).item()
                    if mouth_open_val > mouth_close_val:
                        self.mouth_open_ratio = 0.0
                    else:
                        self.mouth_open_ratio = 1.0
                    print(f"Mouth Open in Image : {self.mouth_open_ratio}")

                self.source_image_dirty = True
                self.update_source_image_bitmap()
                self.Refresh()
                self.Update()
            except:
                self.wx_source_image = None
                self.torch_source_image = None
                message_dialog = wx.MessageDialog(self, "Could not load image " + image_file_name, "Poser", wx.OK)
                message_dialog.ShowModal()
                message_dialog.Destroy()
        file_dialog.Destroy()

    def paint_source_image_panel(self, event: wx.Event):
        wx.BufferedPaintDC(self.source_image_panel, self.source_image_bitmap)

    def paint_result_image_panel(self, event: wx.Event):
        wx.BufferedPaintDC(self.result_image_panel, self.result_image_bitmap)

    def draw_nothing_yet_string_to_bitmap(self, bitmap):
        dc = wx.MemoryDC()
        dc.SelectObject(bitmap)

        dc.Clear()
        font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
        dc.SetFont(font)
        w, h = dc.GetTextExtent("Nothing yet!")
        dc.DrawText("Nothing yet!", (self.image_size - w) // 2, (self.image_size - - h) // 2)

        del dc

    def draw_nothing_yet_string(self, dc):
        dc.Clear()
        font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
        dc.SetFont(font)
        w, h = dc.GetTextExtent("Nothing yet!")
        dc.DrawText("Nothing yet!", (self.poser.get_image_size() - w) // 2, (self.poser.get_image_size() - h) // 2)

    def update_images(self, event: wx.Event):
        if self.torch_source_image is None:
            self.draw_nothing_yet_string_to_bitmap(self.source_image_bitmap)
            self.draw_nothing_yet_string_to_bitmap(self.result_image_bitmap)
            self.source_image_dirty = False
            self.Refresh()
            self.Update()
            return

        if prerendering_count[self.expression_index][2] == PrerenderingMode.Finished:
            self.msg_text.SetLabel("Prerendering finished.")
            return

        pose = [0.0 for i in range(pose_size)]

        if self.expression_index == 0:
            pose[mouth_aaa_index] = self.index_param_0 / 10.0
        elif self.expression_index == 1:
            pose[mouth_iii_index] = self.index_param_0 / 10.0
        elif self.expression_index == 2:
            pose[mouth_uuu_index] = self.index_param_0 / 10.0
        elif self.expression_index == 3:
            pose[mouth_eee_index] = self.index_param_0 / 10.0
        elif self.expression_index == 4:
            pose[mouth_ooo_index] = self.index_param_0 / 10.0
        elif self.expression_index == 5:
            pose[mouth_raised_corner_left_index] = self.index_param_0 / 10.0
            pose[mouth_raised_corner_right_index] = self.index_param_0 / 10.0
        elif self.expression_index == 6:
            pose[mouth_lowered_corner_left_index] = self.index_param_0 / 10.0
            pose[mouth_lowered_corner_right_index] = self.index_param_0 / 10.0
        elif self.expression_index ==7:
            pose[mouth_aaa_index] =self.mouth_open_ratio
            pose[iris_rotation_y_index] = (self.index_param_0 - 5) / 10.0
            pose[iris_rotation_x_index] = (self.index_param_1 - 5) / 10.0
        elif self.expression_index ==8:
            pose[mouth_aaa_index] = self.mouth_open_ratio
            pose[eye_surprised_left_index] = 1.0
            pose[eye_surprised_right_index] = 1.0
            pose[iris_small_left_index] = 0.1
            pose[iris_small_right_index] = 0.1
            pose[iris_rotation_y_index] = (self.index_param_0 - 5) / 10.0
            pose[iris_rotation_x_index] = (self.index_param_1 - 5) / 10.0
        elif self.expression_index == 9:
            pose[mouth_aaa_index] = self.mouth_open_ratio
            smile_degree = self.index_param_1 / 10.0
            pose[eye_wink_left_index] = (1 - smile_degree) * (self.index_param_0 / 10.0)
            pose[eye_wink_right_index] = (1 - smile_degree) * (self.index_param_0 / 10.0)
            pose[eye_happy_wink_left_index] = smile_degree * (self.index_param_0 / 10.0)
            pose[eye_happy_wink_right_index] = smile_degree * (self.index_param_0 / 10.0)
        elif self.expression_index == 10:
            pose[mouth_aaa_index] = self.mouth_open_ratio
            pose[eye_relaxed_left_index] = self.index_param_0 / 10.0
            pose[eye_relaxed_right_index] = self.index_param_0 / 10.0
        elif self.expression_index == 11:
            pose[mouth_aaa_index] = self.mouth_open_ratio
            pose[eye_raised_lower_eyelid_left_index] = self.index_param_0 / 10.0
            pose[eye_raised_lower_eyelid_right_index] = self.index_param_0 / 10.0
        elif self.expression_index == 12:
            pass
        elif self.expression_index == 13:
            pose[eyebrow_troubled_left_index] = self.index_param_0 / 10.0
            pose[eyebrow_troubled_right_index] = self.index_param_0 / 10.0
        elif self.expression_index == 14:
            pose[eyebrow_angry_left_index] = self.index_param_0 / 10.0
            pose[eyebrow_angry_right_index] = self.index_param_0 / 10.0
        elif self.expression_index == 15:
            pose[eyebrow_lowered_left_index] = self.index_param_0 / 10.0
            pose[eyebrow_lowered_right_index] = self.index_param_0 / 10.0
        elif self.expression_index == 16:
            pose[eyebrow_serious_left_index] = self.index_param_0 / 10.0
            pose[eyebrow_serious_right_index] = self.index_param_0 / 10.0
        elif self.expression_index == 17:
            pose[eyebrow_happy_left_index] = self.index_param_0 / 10.0
            pose[eyebrow_happy_right_index] = self.index_param_0 / 10.0
        elif self.expression_index == 18:
            pose[mouth_aaa_index] = self.mouth_open_ratio
            pose[eye_surprised_left_index] = 1.0
            pose[eye_surprised_right_index] = 1.0
        elif self.expression_index == 19:
            pose[mouth_aaa_index] = self.mouth_open_ratio
            pose[head_x_index] = self.index_param_0 * 2.0 -1.0
        elif self.expression_index == 20:
            pose[mouth_aaa_index] = self.mouth_open_ratio
            pose[head_y_index] = self.index_param_0 * 2.0 -1.0
        elif self.expression_index == 21:
            pose[mouth_aaa_index] = self.mouth_open_ratio
            pose[body_y_index] = self.index_param_0 * 2.0 -1.0
        elif self.expression_index == 22:
            pose[mouth_aaa_index] = self.mouth_open_ratio
            pose[neck_z_index] = self.index_param_0 * 2.0 - 1.0
            pose[body_z_index] = (self.index_param_0 * 2.0 - 1.0) * (self.index_param_1 / 5.0 - 1.0)
        elif self.expression_index == 23:
            pose[mouth_aaa_index] = self.mouth_open_ratio
            pose[breathing_index] = self.index_param_0 * 1.0
        elif self.expression_index == 24:
            pass
        else:
            return


        savefilename = self.image_filename_base + "_" + expression_str[self.expression_index] + "_" + format(self.index_param_1, '0=2') + "_" + format(self.index_param_0, '0=2') + ".png"
        savefullpath = os.path.join(self.image_savepath, savefilename)
        if os.path.isfile(savefullpath) == True:
            laveltext = savefilename + " is already exist. Skip..."
            self.msg_text.SetLabel(laveltext)
            self.index_param_0 += 1
            if self.index_param_0 > prerendering_count[self.expression_index][0]:
                self.index_param_0 = 0
                self.index_param_1 += 1
                if self.index_param_1 > prerendering_count[self.expression_index][1]:
                    self.index_param_1 = 0
                    self.expression_index +=1
            return
        else:
            laveltext = "Rendering : " + savefilename
            self.msg_text.SetLabel(laveltext)

        pose_tensor = torch.tensor(pose, device=self.device, dtype=self.dtype)
        if prerendering_count[self.expression_index][2] == PrerenderingMode.BaseMouth:
            with torch.no_grad():
                output_image = self.poser.pose(self.torch_source_image, pose_tensor, 11)[0].detach().cpu()
        elif prerendering_count[self.expression_index][2] == PrerenderingMode.EyeRotation:
            if self.mask_torch_image is None:
                maskpose = [0.0 for i in range(pose_size)]
                maskpose[mouth_aaa_index] = self.mouth_open_ratio
                maskpose[eye_wink_left_index] = 1.0
                maskpose[eye_wink_right_index] = 1.0
                with torch.no_grad():
                    maskpose_tensor = torch.tensor(maskpose, device=self.device, dtype=self.dtype)
                    mask_eye_0 = self.poser.pose(self.torch_source_image, maskpose_tensor, 9)[0].detach().cpu()
                maskpose = [0.0 for i in range(pose_size)]
                maskpose[mouth_aaa_index] = self.mouth_open_ratio
                maskpose[eye_happy_wink_left_index] = 1.0
                maskpose[eye_happy_wink_right_index] = 1.0
                with torch.no_grad():
                    maskpose_tensor = torch.tensor(maskpose, device=self.device, dtype=self.dtype)
                    mask_eye_1 = self.poser.pose(self.torch_source_image, maskpose_tensor, 9)[0].detach().cpu()
                maskpose = [0.0 for i in range(pose_size)]
                maskpose[mouth_aaa_index] = self.mouth_open_ratio
                maskpose[eye_relaxed_left_index] = 1.0
                maskpose[eye_relaxed_right_index] = 1.0
                with torch.no_grad():
                    maskpose_tensor = torch.tensor(maskpose, device=self.device, dtype=self.dtype)
                    mask_eye_2 = self.poser.pose(self.torch_source_image, maskpose_tensor, 9)[0].detach().cpu()
                maskpose[mouth_aaa_index] = self.mouth_open_ratio
                maskpose[eye_surprised_left_index] = 1.0
                maskpose[eye_surprised_right_index] = 1.0
                with torch.no_grad():
                    maskpose_tensor = torch.tensor(maskpose, device=self.device, dtype=self.dtype)
                    mask_eye_3 = self.poser.pose(self.torch_source_image, maskpose_tensor, 12)[0].detach().cpu()
                mask_eye = (mask_eye_0 > 0.2) * 1.0 + (mask_eye_1 > 0.2) * 1.0 + (mask_eye_2 > 0.2) * 1.0 + (mask_eye_3 > 0.2) * 1.0
                self.mask_torch_image = (mask_eye > 0.5) * 2.0 - 1.0
            with torch.no_grad():
                output_image = self.poser.pose(self.torch_source_image, pose_tensor, 11)[0].detach().cpu()
                output_image[3, :, :] = self.mask_torch_image
        elif prerendering_count[self.expression_index][2] == PrerenderingMode.Eyelid:
            with torch.no_grad():
                output_image_temp = self.poser.pose(self.torch_source_image, pose_tensor, 10)[0].detach().cpu()
                output_mask = self.poser.pose(self.torch_source_image, pose_tensor, 9)[0].detach().cpu()
                output_image = output_image_temp
                output_image[3, :, :] = output_mask * 2.0 - 1.0
        elif prerendering_count[self.expression_index][2] == PrerenderingMode.EyebrowMask:
            with torch.no_grad():
                output_image = self.poser.pose(self.torch_source_image, pose_tensor, 26)[0].detach().cpu()
                output_mask = self.poser.pose(self.torch_source_image, pose_tensor, 27)[0].detach().cpu()
                output_image[3, :, :] = output_mask * 2.0 - 1.0
        elif prerendering_count[self.expression_index][2] == PrerenderingMode.Eyebrow:
            with torch.no_grad():
                output_image = self.poser.pose(self.torch_source_image, pose_tensor, 18)[0].detach().cpu()
        elif prerendering_count[self.expression_index][2] == PrerenderingMode.RotationMap:
            with torch.no_grad():
                output_image = self.poser.pose(self.torch_source_image, pose_tensor, 4)[0].detach().cpu()
                map_np = map_np_convert(output_image)
            mapfilename = self.image_filename_base + "_" + expression_str[self.expression_index] + "_" + format(self.index_param_1, '0=2') + "_" + format(self.index_param_0, '0=2') + "_map.npy"
            mapfullpath = os.path.join(self.image_savepath, mapfilename)
            numpy.save(mapfullpath, map_np)

            map_image = map_preview(map_np)
            pil_map = PIL.Image.fromarray(map_image, mode='RGBA')
            mapfilename = self.image_filename_base + "_" + expression_str[self.expression_index] + "_" + format(self.index_param_1, '0=2') + "_" + format(self.index_param_0, '0=2') + "_map.png"
            mapfullpath = os.path.join(self.image_savepath, mapfilename)
            pil_map.save(mapfullpath)
        elif prerendering_count[self.expression_index][2] == PrerenderingMode.SourceImage:
            output_image = self.torch_source_image.detach().cpu()
            


        numpy_image = convert_output_image_from_torch_to_numpy(output_image)
        self.last_output_numpy_image = numpy_image
        wx_image = wx.ImageFromBuffer(
            numpy_image.shape[0],
            numpy_image.shape[1],
            numpy_image[:, :, 0:3].tobytes(),
            numpy_image[:, :, 3].tobytes())
        wx_bitmap = wx_image.ConvertToBitmap()

        dc = wx.MemoryDC()
        dc.SelectObject(self.result_image_bitmap)
        dc.Clear()
        dc.DrawBitmap(wx_bitmap,
                      (self.image_size - numpy_image.shape[0]) // 2,
                      (self.image_size - numpy_image.shape[1]) // 2,
                      True)
        del dc

        self.Refresh()
        self.Update()

        pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA')

        pil_image.save(savefullpath)

        self.index_param_0 += 1
        if self.index_param_0 > prerendering_count[self.expression_index][0]:
            self.index_param_0 = 0
            self.index_param_1 += 1
            if self.index_param_1 > prerendering_count[self.expression_index][1]:
                self.index_param_1 = 0
                self.expression_index +=1

        return


    def save_image(self):
        if self.last_output_numpy_image is None:
            logging.info("There is no output image to save!!!")
            return

        dir_name = "data/images"
        file_dialog = wx.FileDialog(self, "Choose an image", dir_name, "", "*.png", wx.FD_SAVE)
        if file_dialog.ShowModal() == wx.ID_OK:
            image_file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
            try:
                if os.path.exists(image_file_name):
                    message_dialog = wx.MessageDialog(self, f"Override {image_file_name}", "Manual Poser",
                                                      wx.YES_NO | wx.ICON_QUESTION)
                    result = message_dialog.ShowModal()
                    if result == wx.ID_YES:
                        self.save_last_numpy_image(image_file_name)
                    message_dialog.Destroy()
                else:
                    self.save_last_numpy_image(image_file_name)
            except:
                message_dialog = wx.MessageDialog(self, f"Could not save {image_file_name}", "Manual Poser", wx.OK)
                message_dialog.ShowModal()
                message_dialog.Destroy()
        file_dialog.Destroy()

    def save_last_numpy_image(self, image_file_name):
        numpy_image = self.last_output_numpy_image
        pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA')
        os.makedirs(os.path.dirname(image_file_name), exist_ok=True)
        pil_image.save(image_file_name)

    def create_timers(self):
        self.animation_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_images, id=self.animation_timer.GetId())

    def on_close(self, event: wx.Event):
        # Stop the timers
        self.animation_timer.Stop()

        # Destroy the windows
        self.Destroy()
        event.Skip()

    @staticmethod
    def convert_to_100(x):
        return int(max(0.0, min(1.0, x)) * 100)

    def update_source_image_bitmap(self):
        dc = wx.MemoryDC()
        dc.SelectObject(self.source_image_bitmap)
        if self.wx_source_image is None:
            self.draw_nothing_yet_string(dc)
        else:
            dc.Clear()
            dc.DrawBitmap(self.wx_source_image, 0, 0, True)
            if self.source_image_string is None:
                pass
            else:
                font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
                dc.SetFont(font)
                w, h = dc.GetTextExtent(self.source_image_string)
                dc.DrawText(self.source_image_string, (self.poser.get_image_size() - w) // 2, h)

        del dc

    def blend_with_background(self, numpy_image, background):
        alpha = numpy_image[3:4, :, :]
        color = numpy_image[0:3, :, :]
        new_color = color * alpha + (1.0 - alpha) * background[0:3, :, :]
        return torch.cat([new_color, background[3:4, :, :]], dim=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Control characters with movement captured by iFacialMocap.')
    parser.add_argument(
        '--model',
        type=str,
        required=False,
        default='standard_float',
        choices=['standard_float', 'separable_float', 'standard_half', 'separable_half'],
        help='The model to use.')
    parser.add_argument(
        '--timer',
        type=int,
        required=False,
        default=20,
        # choices=range(5, 2000),
        help='Animation cycle ; 5-2000[ms].')
    args = parser.parse_args()

    # device = torch.device('cuda')
    # device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        poser = load_poser(args.model, device)
    except RuntimeError as e:
        print(e)
        sys.exit()

    # preload models

    default_pose = torch.zeros(45).to(device).to(poser.get_dtype())
    poser.pose(torch.zeros(4, 512, 512).to(device).to(poser.get_dtype()), default_pose)[0].float()

    app = wx.App()
    main_frame = MainFrame(poser, device)
    main_frame.Show(True)
    # main_frame.capture_timer.Start(10)
    maintimer = args.timer
    if maintimer < 5:
        print("--timer value is too small !!!")
        maintimer = 5
    elif maintimer > 2000:
        print("--timer value is too large !!!")
        maintimer = 2000
    print(f"Animation timer cycle is {maintimer} [ms].")
    main_frame.animation_timer.Start(maintimer)
    app.MainLoop()
