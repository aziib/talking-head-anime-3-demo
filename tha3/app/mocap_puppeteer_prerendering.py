import argparse
import os
import socket
import sys
import threading
import time
from typing import Optional, Dict, List

sys.path.append(os.getcwd())

from tha3.mocap.ifacialmocap_pose import create_default_ifacialmocap_pose
from tha3.mocap.ifacialmocap_v2 import IFACIALMOCAP_PORT, IFACIALMOCAP_START_STRING, parse_ifacialmocap_v2_pose, \
    parse_ifacialmocap_v1_pose, parse_meowface_pose, parse_vmc_pose, parse_vts_pose, parse_vmc_pose_list, parse_vmc_perfectsync_pose_list
# from tha3.poser.modes.load_poser import load_poser

from tha3.mocap.ifacialmocap_poser_converter_25 import SmartPhoneApp
from tha3.mocap.ifacialmocap_poser_converter_25 import IFacialMocapPoseConverter25Args

import wx
import wx.adv
# from torch.nn import functional as F
import numpy
from numba import jit
import math
import scipy.optimize
import json
import PIL.Image

# import pdb

# from tha3.poser.poser import Poser
from tha3.mocap.ifacialmocap_constants import *
from tha3.mocap.ifacialmocap_pose_converter import IFacialMocapPoseConverter
from tha3.util import torch_linear_to_srgb, resize_PIL_image, extract_PIL_image_from_filelike, \
    extract_pytorch_image_from_PIL_image, numpy_srgb_to_linear


import tha3.mocap.ifacialmocap_add as ifadd

from tha3.poser.modes.pose_parameters import get_pose_parameters
from enum import Enum, IntEnum

class EyebrowDownMode(Enum):
    TROUBLED = 1
    ANGRY = 2
    LOWERED = 3
    SERIOUS = 4

class WinkMode(Enum):
    NORMAL = 1
    RELAXED = 2

def clamp(x, min_value, max_value):
    return max(min_value, min(max_value, x))

class FpsStatistics:
    def __init__(self):
        self.count = 100
        self.fps = []

    def add_fps(self, fps):
        self.fps.append(fps)
        while len(self.fps) > self.count:
            del self.fps[0]

    def get_average_fps(self):
        if len(self.fps) == 0:
            return 0.0
        else:
            return sum(self.fps) / len(self.fps)

max_expresson_index = 25

expression_layer_count = 14

class PoseLayer(IntEnum):
    Source = 0
    Mouth = 1
    EyeLeft = 2
    EyeRight = 3
    EyelidLeft = 4
    EyelidRight = 5
    EyebrowMask = 6
    EyebrowLeft = 7
    EyebrowRight = 8
    HeadX = 9
    HeadY = 10
    BodyY = 11
    BodyNeckZ = 12
    Breath = 13

class ExpressionChoice(IntEnum):
    BaseAAA = 0
    BaseIII = 1
    BaseUUU = 2
    BaseEEE = 3
    BaseOOO = 4
    BaseRaised = 5
    BaseLowered = 6
    EyeNormal = 7
    EyeSurprised = 8
    EyelidNormalHappy = 9
    EyelidRelaxed = 10
    EyelidLower = 11
    EyebrowMask = 12
    EyebrowTroubled = 13
    EyebrowAngry = 14
    EyebrowLowered = 15
    EyebrowSerious = 16
    EyebrowHappy = 17
    EyelidSurprised = 18
    FlowmapX = 19
    FlowmapHeadY = 20
    FlowmapBodyY = 21
    FlowmapZ = 22
    FlowmapBreath = 23
    FullImage = 24
    Finished = 25

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
    [10, 0, PrerenderingMode.BaseMouth, 192], # 0
    [10, 0, PrerenderingMode.BaseMouth, 192], # 1
    [10, 0, PrerenderingMode.BaseMouth, 192], # 2
    [10, 0, PrerenderingMode.BaseMouth, 192], # 3
    [10, 0, PrerenderingMode.BaseMouth, 192], # 4
    [10, 0, PrerenderingMode.BaseMouth, 192], # 5
    [10, 0, PrerenderingMode.BaseMouth, 192], # 6
    [10, 10, PrerenderingMode.EyeRotation, 192], # 7
    [10, 10, PrerenderingMode.EyeRotation, 192], # 8
    [10, 10, PrerenderingMode.Eyelid, 192], # 9
    [10, 0, PrerenderingMode.Eyelid, 192], # 10
    [10, 0, PrerenderingMode.Eyelid, 192], # 11
    [0, 0, PrerenderingMode.EyebrowMask, 128], # 12
    [10, 0, PrerenderingMode.Eyebrow, 128], # 13
    [10, 0, PrerenderingMode.Eyebrow, 128], # 14
    [10, 0, PrerenderingMode.Eyebrow, 128], # 15
    [10, 0, PrerenderingMode.Eyebrow, 128], # 16
    [10, 0, PrerenderingMode.Eyebrow, 128], # 17
    [0, 0, PrerenderingMode.Eyelid, 192], # 18
    [1, 0, PrerenderingMode.RotationMap, 512], # 19
    [1, 0, PrerenderingMode.RotationMap, 512], # 20
    [1, 0, PrerenderingMode.RotationMap, 512], # 21
    [1, 10, PrerenderingMode.RotationMap, 512], # 22
    [1, 0, PrerenderingMode.RotationMap, 512], # 23
    [0, 0, PrerenderingMode.SourceImage, 512], # 24
    [0, 0, PrerenderingMode.Finished, 512], # 25
]

def load_prerendering_images(path, filename):
    expression_index = 0
    index_param_0 = 0
    index_param_1 = 0

    image_size = 512

    image_missing = False

    loadimage = [None] * max_expresson_index
    for expression_index in range(max_expresson_index):
        loadimage[expression_index] = [None] * (prerendering_count[expression_index][1] + 1)
        for index_param_1 in range(prerendering_count[expression_index][1] + 1):
            loadimage[expression_index][index_param_1] = [None] * (prerendering_count[expression_index][0] + 1)
            for index_param_0 in range(prerendering_count[expression_index][0] + 1):
                if prerendering_count[expression_index][2] == PrerenderingMode.RotationMap:
                    loadfilename = filename + "_" + expression_str[expression_index] + "_" + format(index_param_1, '0=2') + "_" + format(index_param_0, '0=2') + "_map.npy"
                    loadfullpath = os.path.join(path, loadfilename)
                    if os.path.isfile(loadfullpath) == True:
                        # print("loading " + loadfullpath)
                        np_image_map_l = numpy.load(loadfullpath)
                        loadimage[expression_index][index_param_1][index_param_0] = np_image_map_l
                        pass
                    else:
                        print(loadfilename + " not found!")
                        image_missing = True
                        loadimage[expression_index][index_param_1][index_param_0] = numpy.zeros((image_size, image_size, 2), dtype = numpy.float32)
                        pass
                else:
                    loadfilename = filename + "_" + expression_str[expression_index] + "_" + format(index_param_1, '0=2') + "_" + format(index_param_0, '0=2') + ".png"
                    loadfullpath = os.path.join(path, loadfilename)
                    if os.path.isfile(loadfullpath) == True:
                        # print("loading " + loadfullpath)
                        pil_image_temp = extract_PIL_image_from_filelike(loadfullpath)
                        np_image_temp = numpy.asarray(pil_image_temp)
                        loadimage[expression_index][index_param_1][index_param_0] = np_image_temp
                        pass
                    else:
                        print(loadfilename + " not found!")
                        image_missing = True
                        loadimage[expression_index][index_param_1][index_param_0] = \
                            numpy.zeros((prerendering_count[expression_index][3], prerendering_count[expression_index][3], 4), dtype = numpy.uint8)
                        pass

    return loadimage, image_missing

@jit(nopython=True)
def blend_numpy_image(foreground, background):
    new_color = background[ :, :, :]
    alpha = foreground[ :, :, 3:4] / 255.0
    color = foreground[ :, :, 0:3]
    new_color[ :, :, 0:3] = 0.5 + color * alpha + (1.0 - alpha) * background[ :, :, 0:3]
    #     return numpy.concatenate([new_color, background[ :, :, 3:4]], axis=2)
    new_color = new_color.astype(numpy.uint8)
    return new_color

# @jit(nopython=False)
def comp_expression_image(pose, imageset):
    # output_image = numpy.zeros((2048, 2048, 4), dtype = numpy.float64)
    output_image = numpy.zeros((512, 512, 4), dtype = numpy.uint8)

    image_Source       = imageset[ExpressionChoice.FullImage][0][0]
    image_Mouth        = imageset[pose[PoseLayer.Mouth       ][0]][0][pose[PoseLayer.Mouth][1]]
    image_EyeLeft      = imageset[pose[PoseLayer.EyeLeft     ][0]][pose[PoseLayer.EyeLeft    ][2]][pose[PoseLayer.EyeLeft    ][1]]
    image_EyeRight     = imageset[pose[PoseLayer.EyeRight    ][0]][pose[PoseLayer.EyeRight   ][2]][pose[PoseLayer.EyeRight   ][1]]
    image_EyelidLeft   = imageset[pose[PoseLayer.EyelidLeft  ][0]][pose[PoseLayer.EyelidLeft ][2]][pose[PoseLayer.EyelidLeft ][1]]
    image_EyelidRight  = imageset[pose[PoseLayer.EyelidRight ][0]][pose[PoseLayer.EyelidRight][2]][pose[PoseLayer.EyelidRight][1]]
    image_EyebrowMask  = imageset[pose[PoseLayer.EyebrowMask ][0]][0][0]
    image_EyebrowLeft  = imageset[pose[PoseLayer.EyebrowLeft ][0]][0][pose[PoseLayer.EyebrowLeft ][1]]
    image_EyebrowRight = imageset[pose[PoseLayer.EyebrowRight][0]][0][pose[PoseLayer.EyebrowRight][1]]
    # map_HeadX = None
    # map_HeadY = None
    # map_BodyY = None
    # map_BodyNeckZ = None
    # map_Breath = None

    output_image[:, :, :] = image_Source[:, :, :]
    # output_image[ :, :, :] = [0, 0, 0, 255]
    output_image[32:224, 160:352, :] = image_Mouth[:, :, :]
    output_image[32:224, 256:352, :] = blend_numpy_image(     image_EyeLeft[:,  96:192, :], output_image[ 32:224, 256:352, :])
    output_image[32:224, 160:256, :] = blend_numpy_image(    image_EyeRight[:,   0: 96, :], output_image[ 32:224, 160:256, :])
    output_image[32:224, 256:352, :] = blend_numpy_image(  image_EyelidLeft[:,  96:192, :], output_image[ 32:224, 256:352, :])
    output_image[32:224, 160:256, :] = blend_numpy_image( image_EyelidRight[:,   0: 96, :], output_image[ 32:224, 160:256, :])
    output_image[64:192, 192:320, :] = blend_numpy_image( image_EyebrowMask[:,    :   , :], output_image[ 64:192, 192:320, :])
    output_image[64:192, 256:320, :] = blend_numpy_image( image_EyebrowLeft[:,  64:128, :], output_image[ 64:192, 256:320, :])
    output_image[64:192, 192:256, :] = blend_numpy_image(image_EyebrowRight[:,   0: 64, :], output_image[ 64:192, 192:256, :])

    # output_image = output_image.astype(numpy.uint8)
    return output_image

# @jit(nopython=False)
def comp_map_image(pose, imageset):
    image_size = 512
    large_size = 512
    comp_map = numpy.zeros((large_size, large_size, 2), dtype = numpy.float32)
    # output_image = numpy.zeros((large_size, large_size, 4), dtype = numpy.float64)
    # large_output = numpy.zeros((large_size, large_size, 3), dtype = numpy.int32)

    map_HeadX     = imageset[ExpressionChoice.FlowmapX     .value][pose[PoseLayer.HeadX    .value][2]][pose[PoseLayer.HeadX    .value][1]]
    map_HeadY     = imageset[ExpressionChoice.FlowmapHeadY .value][pose[PoseLayer.HeadY    .value][2]][pose[PoseLayer.HeadY    .value][1]]
    map_BodyY     = imageset[ExpressionChoice.FlowmapBodyY .value][pose[PoseLayer.BodyY    .value][2]][pose[PoseLayer.BodyY    .value][1]]
    map_BodyNeckZ = imageset[ExpressionChoice.FlowmapZ     .value][pose[PoseLayer.BodyNeckZ.value][2]][pose[PoseLayer.BodyNeckZ.value][1]]
    map_Breath    = imageset[ExpressionChoice.FlowmapBreath.value][pose[PoseLayer.Breath   .value][2]][pose[PoseLayer.Breath   .value][1]]

    value_HeadX     = pose[PoseLayer.HeadX    .value][0] / 1024.0
    value_HeadY     = pose[PoseLayer.HeadY    .value][0] / 1024.0
    value_BodyY     = pose[PoseLayer.BodyY    .value][0] / 1024.0
    value_BodyNeckZ = pose[PoseLayer.BodyNeckZ.value][0] / 1024.0
    value_Breath    = pose[PoseLayer.Breath   .value][0] / 1024.0

    comp_map[:, :, :] = (map_HeadX    [:, :, 0:2]) * value_HeadX \
                       + (map_HeadY    [:, :, 0:2]) * value_HeadY \
                       + (map_BodyY    [:, :, 0:2]) * value_BodyY \
                       + (map_BodyNeckZ[:, :, 0:2]) * value_BodyNeckZ \
                       + (map_Breath   [:, :, 0:2]) * value_Breath

    # np_output = numpy.zeros((large_size, large_size, 3), dtype = numpy.int32)
    # np_output[:, :, 0:2] = comp_map[:, :, :]
    # np_output[:, :, 2:3] = 1
    # large_output = np_output
    return comp_map

@jit(nopython=True)
def transform_image_with_map(source_image, np_image_map, image_size = 512):
    np_output = numpy.zeros((image_size, image_size, 2), dtype = numpy.float32)
    # souce_addr = numpy.zeros((image_size, image_size, 2), dtype = numpy.uint32)
    output_flag = numpy.zeros((image_size, image_size, 1), dtype = numpy.uint8)
    target_map = numpy.zeros((image_size + 1, image_size + 1, 2), dtype = numpy.uint32)
    output_mask = numpy.zeros((image_size, image_size, 1), dtype = numpy.uint8)
    output_image = numpy.zeros((image_size, image_size, 4), dtype = numpy.uint32)
    # np_temp = 65535

    for y in range(image_size):
        for x in range(image_size):
            target_map[y, x, 0] = min(image_size - 1, max(0, y - round(np_image_map[y, x, 0])))
            target_map[y, x, 1] = min(image_size - 1, max(0, x - round(np_image_map[y, x, 1])))
        target_map[y, image_size, 0] = min(image_size - 1, max(0, y - round(np_image_map[y, image_size - 1, 0])))
        target_map[y, image_size, 1] = image_size - 1
    for x in range(image_size + 1):
        target_map[image_size, x, 0] = image_size - 1
        target_map[image_size, x, 1] = min(image_size - 1, max(0, x - round(np_image_map[image_size - 1, x, 1])))

    for y in range(image_size):
        for x in range(image_size):
            if source_image[y, x][3] == 0:
                pass
            else:
                target_y = target_map[y, x, 0]
                target_x = target_map[y, x, 1]
                # target_y = min(image_size - 1, max(0, y + round(map[y, x][0])))
                # target_x = min(image_size - 1, max(0, x + round(map[y, x][1])))
                np_output[target_y, target_x][:] = np_output[target_y, target_x][:] + np_image_map[y, x][0:2]
                # souce_addr[target_y, target_x][:] = souce_addr[target_y, target_x][:] + [x, y]
                mask_x_low = min(target_map[y, x, 1], target_map[y, x + 1, 1], target_map[y + 1, x, 1], target_map[y + 1, x + 1, 1])
                mask_x_high = max(target_map[y, x, 1], target_map[y, x + 1, 1], target_map[y + 1, x, 1], target_map[y + 1, x + 1, 1])
                mask_y_low = min(target_map[y, x, 0], target_map[y, x + 1, 0], target_map[y + 1, x, 0], target_map[y + 1, x + 1, 0])
                mask_y_high = max(target_map[y, x, 0], target_map[y, x + 1, 0], target_map[y + 1, x, 0], target_map[y + 1, x + 1, 0])
                if mask_x_low == mask_x_high:
                    mask_x_high +=1
                if mask_y_low == mask_y_high:
                    mask_y_low += 1
                output_mask[mask_y_low:mask_y_high, mask_x_low:mask_x_high, 0] = 1
                output_image[mask_y_low:mask_y_high, mask_x_low:mask_x_high][:] = output_image[mask_y_low:mask_y_high, mask_x_low:mask_x_high][:] + source_image[y, x][:]
                output_flag[mask_y_low:mask_y_high, mask_x_low:mask_x_high, 0] = output_flag[mask_y_low:mask_y_high, mask_x_low:mask_x_high, 0] + 1
    
    for y in range(0, image_size - 1):
        for x in range(0, image_size - 1):
            if output_flag[y, x, 0] == 0:
                output_flag[y, x, 0] = 1

    output_image = output_image / output_flag

    return output_image

class MainFrame(wx.Frame):
    def __init__(self, pose_converter: IFacialMocapPoseConverter, mocap_port = 49983):
#        super().__init__(None, wx.ID_ANY, "iFacialMocap Puppeteer (Marigold)")
        super().__init__(None, wx.ID_ANY, "Face Motion Capture Puppeteer (Talking Head Anime 3) - Prerendering")
        self.pose_converter = pose_converter
        self.args = IFacialMocapPoseConverter25Args()

        self.mocap_port = mocap_port

        self.resized_image_size = 512
        self.image_size = 512
        self.image_view_size = 512

        self.ifacialmocap_pose = create_default_ifacialmocap_pose()
        self.source_image_bitmap = wx.Bitmap(self.image_view_size, self.image_view_size)
        self.result_image_bitmap = wx.Bitmap(self.image_view_size, self.image_view_size)
        self.wx_source_image = None
        self.last_pose = None
        self.fps_statistics = FpsStatistics()
        self.last_update_time = None
        self.same_pose_count = 0

        self.np_source_large = None
        self.source_image_string = None

        self.prerendering_numpy_image = None

        self.max_thread_count = 1
        self.result_image = [None] * self.max_thread_count
        self.result_image_updated = [False] * self.max_thread_count
        self.calc_image_number = 0
        self.show_image_number = 0
        self.image_queue_count = 0
        self.image_prerendering = None

        self.last_pose_layer = numpy.zeros((expression_layer_count, 3), dtype = numpy.int64)

        # self.mapbase = numpy.zeros((512, 512, 2), dtype = numpy.int64)
        # for y in range(512):
        #     for x in range(512):
        #         self.mapbase[x, y, 1] = x
        #         self.mapbase[x, y, 0] = y

        self.last_show_index = -1
        self.last_output_index = -1

        self.vts_already_request = False
        self.vts_ip = "192.168.0.1"
        self.vts_port = 21412

        self.vmc_cache = []
        self.poseIsPerfectsync = False

        self.create_receiving_socket()
        self.create_ui()
        self.create_timers()
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.read_config_file()

        self.update_source_image_bitmap()
        self.update_result_image_bitmap()

    def create_receiving_socket(self):
        self.receiving_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.receiving_socket.bind(("", IFACIALMOCAP_PORT))
        self.receiving_socket.bind(("", self.mocap_port))
        self.receiving_socket.setblocking(False)

    def create_timers(self):
        self.capture_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_capture_panel, id=self.capture_timer.GetId())
        self.animation_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_result_image_bitmap, id=self.animation_timer.GetId())

    def read_config_file(self):
        # config_json_dict = {}

        if os.path.exists('tha3sw_config.json'):
            try:
                with open('tha3sw_config.json') as f:
                        config_json_dict = json.load(f)
            except Exception as e:# json.JSONDecodeError:
                print(e)
                config_json_dict = {}
        else:
            print("JSON file is not found!")
            config_json_dict = {}

        save_config = config_json_dict.get("save_config", None)
        if save_config == True:
            capture_ip = config_json_dict.get("capture_ip", "192.168.0.1")
            mocap_method = config_json_dict.get("mocap_method", 2)
            eyebrow_mode = config_json_dict.get("eyebrow_mode", 0)
            wink_mode = config_json_dict.get("wink_mode", 0)
            # irissize_left = config_json_dict.get("irissize_left", 0)
            # irissize_right = config_json_dict.get("irissize_right", 0)
            # irissize_link = config_json_dict.get("irissize_link", True)
            breathing = config_json_dict.get("breathing", 20)
            head_x = config_json_dict.get("head_x", 0.0)
            head_y = config_json_dict.get("head_y", 0.0)
            neck_z = config_json_dict.get("neck_z", 0.0)
            body_y = config_json_dict.get("body_y", 0.0)
            body_z = config_json_dict.get("body_z", 0.0)
            backgroud = config_json_dict.get("backgroud", 0)

            image_list = config_json_dict.get("image_list_pre", [])
            image_select = config_json_dict.get("image_select_pre", -1)
            image_output = config_json_dict.get("image_output_pre", -1)

            self.capture_device_ip_text_ctrl.SetValue(capture_ip)

            self.pose_converter.sp_app_choice.SetSelection(mocap_method)
            self.pose_converter.change_sp_app(self)
            self.pose_converter.eyebrow_down_mode_choice.SetSelection(eyebrow_mode)
            self.pose_converter.change_eyebrow_down_mode(self)
            self.pose_converter.wink_mode_choice.SetSelection(wink_mode)
            self.pose_converter.change_wink_mode(self)

            # self.pose_converter.iris_left_slider.SetValue(irissize_left)
            # self.pose_converter.iris_right_slider.SetValue(irissize_right)
            # self.pose_converter.link_left_right_irises.SetValue(irissize_link)
            # self.pose_converter.link_left_right_irises_clicked(self)
            # self.pose_converter.change_iris_size(self)

            self.pose_converter.breathing_frequency_slider.SetValue(breathing)

            ifadd.CAL_HEAD_X = float(head_x)
            ifadd.CAL_HEAD_Y = float(head_y)
            ifadd.CAL_HEAD_Z = float(neck_z)
            ifadd.CAL_BODY_Y = float(body_y)
            ifadd.CAL_BODY_Z = float(body_z)

            self.pose_converter.calibrate_head_x_slider.SetValue(int(ifadd.CAL_HEAD_X))
            self.pose_converter.calibrate_head_y_slider.SetValue(int(ifadd.CAL_HEAD_Y))
            self.pose_converter.calibrate_head_z_slider.SetValue(int(ifadd.CAL_HEAD_Z))

            self.pose_converter.calibrate_body_y_slider.SetValue(int(ifadd.CAL_BODY_Y))
            self.pose_converter.calibrate_body_z_slider.SetValue(int(ifadd.CAL_BODY_Z))

            self.output_background_choice.SetSelection(backgroud)
            self.background_changed(self)

            image_list_index = len(image_list)
            if image_list_index > 0:
                for l in range(image_list_index):

                    image_file_name = image_list[l]
                    image_name = ""
                    image_filename_base = ""
                    image_loadpath = ""
                    if os.path.exists(image_file_name):
                        try:
                            image_name = os.path.basename(image_file_name)
                            image_filepath_base = os.path.dirname(image_file_name)
                            image_filename_base = os.path.splitext(image_name)[0]
                            image_loadpath = os.path.join(image_filepath_base, image_filename_base + "_tha")
                            pil_image = resize_PIL_image(
                                extract_PIL_image_from_filelike(image_file_name),
                                (self.image_size, self.image_size))
                            w, h = pil_image.size
                        except Exception as e:
                            print(e)
                            image_name = image_name + "  (Image Loading Error!)"
                            if image_loadpath == "":
                                image_loadpath = "_tha"
                            w, h = self.image_size, self.image_size
                            pil_image = PIL.Image.new("RGBA", (w, h), (0, 0, 0, 0))
                    else:
                        image_name = image_name + "  (Image Not Found!)"
                        if image_loadpath == "":
                            image_loadpath = "_tha"
                        w, h = self.image_size, self.image_size
                        pil_image = PIL.Image.new("RGBA", (w, h), (0, 0, 0, 0))
                    wx_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                    np_image_large = numpy.asarray(pil_image)
                    if os.path.isdir(image_loadpath) == False:
                        image_name = image_name + "  (Prerendering Images Not Found!)"
                    prerendering_image, image_missing = load_prerendering_images(image_loadpath, image_filename_base)
                    if image_missing == True:
                        image_name = image_name + "  (Some Prerendering Images Are Missing!)"
                    self.source_image_list.Append(image_name, [pil_image, wx_image, np_image_large, prerendering_image, image_file_name])

                if image_list_index <= image_select:
                    image_select = -1
                if image_list_index <= image_output:
                    image_output = -1
                if image_output >= 0:
                    if image_select < 0:
                        image_select = image_output
                    self.last_output_index = image_output
                    image_sets = self.source_image_list.GetClientData(image_output)
                    self.prerendering_numpy_image = image_sets[3]
                    self.np_source_large = image_sets[2]
                    self.last_pose = None
                if image_select >= 0:
                    self.source_image_list.SetSelection(image_select)
                    self.last_show_index = image_select
                    image_sets = self.source_image_list.GetClientData(image_select)
                    self.source_image_string = None
                    self.wx_source_image = image_sets[1]
                # self.update_source_image_bitmap()
                # self.update_result_image_bitmap()
                # self.Refresh()
                tip = wx.adv.RichToolTip("Click to preview image.\nDoubleClick to output animation.", "") # "Quick Guide"
                tip.SetTimeout(10000, 2000)
                tip.ShowFor(self.source_image_list)

        else:
            pass

        return

    def save_config_file(self):
        config_json_dict = {}
        try:
            with open('tha3sw_config.json') as f:
                try:
                    config_json_dict = json.load(f)
                except json.JSONDecodeError:
                    config_json_dict = {}

            save_config = config_json_dict.get("save_config", None)
            if save_config == True:
                capture_ip = self.capture_device_ip_text_ctrl.GetValue()
                mocap_method = self.pose_converter.sp_app_choice.GetSelection()
                eyebrow_mode = self.pose_converter.eyebrow_down_mode_choice.GetSelection()
                wink_mode = self.pose_converter.wink_mode_choice.GetSelection()
                # irissize_left = self.pose_converter.iris_left_slider.GetValue()
                # irissize_right = self.pose_converter.iris_right_slider.GetValue()
                # irissize_link = self.pose_converter.link_left_right_irises.GetValue()
                breathing = self.pose_converter.breathing_frequency_slider.GetValue()
                head_x = ifadd.CAL_HEAD_X
                head_y = ifadd.CAL_HEAD_Y
                neck_z = ifadd.CAL_HEAD_Z
                body_y = ifadd.CAL_BODY_Y
                body_z = ifadd.CAL_BODY_Z
                backgroud = self.output_background_choice.GetSelection()

                image_list_index = self.source_image_list.GetCount()
                image_list = []
                for l in range(image_list_index):
                    image_sets = self.source_image_list.GetClientData(l)
                    image_fullpath = image_sets[4]
                    image_list.append(image_fullpath)
                image_select = self.last_show_index
                image_output = self.last_output_index

                config_json_dict["capture_ip"] = capture_ip
                config_json_dict["mocap_method"] = mocap_method
                config_json_dict["eyebrow_mode"] = eyebrow_mode
                config_json_dict["wink_mode"] = wink_mode
                # config_json_dict["irissize_left"] = irissize_left
                # config_json_dict["irissize_right"] = irissize_right
                # config_json_dict["irissize_link"] = irissize_link
                config_json_dict["breathing"] = breathing
                config_json_dict["head_x"] = head_x
                config_json_dict["head_y"] = head_y
                config_json_dict["neck_z"] = neck_z
                config_json_dict["body_y"] = body_y
                config_json_dict["body_z"] = body_z
                config_json_dict["backgroud"] = backgroud

                config_json_dict["image_list_pre"] = image_list
                config_json_dict["image_select_pre"] = image_select
                config_json_dict["image_output_pre"] = image_output

                with open('tha3sw_config.json', 'w') as f:
                    json.dump(config_json_dict, f, indent=4)
        except Exception as e:
            print (e)

    def on_close(self, event: wx.Event):
        # Stop the timers
        self.animation_timer.Stop()
        self.capture_timer.Stop()

        # Save config file
        self.save_config_file()

        # Close receiving socket
        self.receiving_socket.close()

        # Destroy the windows
        self.Destroy()
        event.Skip()

    def on_start_capture(self, event: wx.Event):
        if ifadd.SP_APP_MODE == SmartPhoneApp.VTUBESTUDIO:
            VTS_PORT = 21412
            capture_device = self.capture_device_ip_text_ctrl.GetValue()
            self.vts_ip = capture_device
            self.vts_port = VTS_PORT
            self.vts_already_request = True
            self.vts_send_request()
        else:
            capture_device_ip_address = self.capture_device_ip_text_ctrl.GetValue()
            out_socket = None
            try:
                address = (capture_device_ip_address, IFACIALMOCAP_PORT)
                out_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                out_socket.sendto(IFACIALMOCAP_START_STRING, address)
            except Exception as e:
                message_dialog = wx.MessageDialog(self, str(e), "Error!", wx.OK)
                message_dialog.ShowModal()
                message_dialog.Destroy()
            finally:
                if out_socket is not None:
                    out_socket.close()

    def vts_send_request(self):
        out_socket = None
        capture_device_ip_address = self.vts_ip
        capture_device_port = self.vts_port
        VTS_START_STRING = '{"messageType":"iOSTrackingDataRequest","time":10.0,"sentBy":"THA3SW","ports":['.encode('utf-8') + f'{self.mocap_port}'.encode('utf-8') + ']}'.encode('utf-8')

        try:
            address = (capture_device_ip_address, capture_device_port)
            out_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            out_socket.sendto(VTS_START_STRING, address)
        except Exception as e:
            self.vts_already_request = False
            message_dialog = wx.MessageDialog(self, str(e), "Error!", wx.OK)
            message_dialog.ShowModal()
            message_dialog.Destroy()
        finally:
            if out_socket is not None:
                out_socket.close()

    def read_ifacialmocap_pose(self):
        if not self.animation_timer.IsRunning():
            self.show_status_indicator(False, False)
            return self.ifacialmocap_pose
        socket_bytes = None
        count = 0
        socket_list = []
        while True:
            try:
                socket_bytes = self.receiving_socket.recv(8192)
                count += 1
                socket_list.append(socket_bytes)
            except socket.error as e:
                break
        if socket_bytes is not None:
            socket_string = socket_bytes.decode("utf-8","ignore")
            pose_cache = self.ifacialmocap_pose
            isPerfectsync = self.poseIsPerfectsync
            try:
                # For debug, please see the following string
                # pdb.set_trace()
                # print(socket_string)
                if ifadd.SP_APP_MODE == SmartPhoneApp.IPHONE:
                    # SmartPhoneApp.IPHONE
                    parse_pose_temp = parse_ifacialmocap_v2_pose(socket_string)
                    if parse_pose_temp[1] == True:
                        self.ifacialmocap_pose = parse_pose_temp[0]
                        self.poseIsPerfectsync = True
                    else:
                        self.ifacialmocap_pose = pose_cache
                        self.poseIsPerfectsync = isPerfectsync
                elif ifadd.SP_APP_MODE == SmartPhoneApp.ANDROID:
                    # SmartPhoneApp.ANDROID
                    parse_pose_temp = parse_meowface_pose(socket_string)
                    if parse_pose_temp[1] == True:
                        self.ifacialmocap_pose = parse_pose_temp[0]
                        self.poseIsPerfectsync = True
                    else:
                        self.ifacialmocap_pose = pose_cache
                        self.poseIsPerfectsync = isPerfectsync
                elif ifadd.SP_APP_MODE == SmartPhoneApp.VMC:
                    socket_cache = self.vmc_cache + socket_list
                    parse_pose_temp = parse_vmc_pose_list(socket_cache, pose_cache)
                    if parse_pose_temp[1] == True:
                        self.ifacialmocap_pose = parse_pose_temp[0]
                        self.poseIsPerfectsync = False
                    else:
                        self.ifacialmocap_pose = pose_cache
                        self.poseIsPerfectsync = isPerfectsync
                    if parse_pose_temp[2] == True:
                        self.vmc_cache = socket_list
                    else:
                        cache_len = len(socket_cache)
                        if cache_len > 2048:
                            self.vmc_cache = socket_cache[cache_len - 2048:]
                        else:
                            self.vmc_cache = socket_cache
                elif ifadd.SP_APP_MODE == SmartPhoneApp.VMCPERFECTSYNC:
                    socket_cache = self.vmc_cache + socket_list
                    parse_pose_temp = parse_vmc_perfectsync_pose_list(socket_cache, pose_cache)
                    if parse_pose_temp[1] == True:
                        self.ifacialmocap_pose = parse_pose_temp[0]
                        self.poseIsPerfectsync = True
                    else:
                        self.ifacialmocap_pose = pose_cache
                        self.poseIsPerfectsync = isPerfectsync
                    if parse_pose_temp[2] == True:
                        self.vmc_cache = socket_list
                    else:
                        cache_len = len(socket_cache)
                        if cache_len > 2048:
                            self.vmc_cache = socket_cache[cache_len - 2048:]
                        else:
                            self.vmc_cache = socket_cache
                elif ifadd.SP_APP_MODE == SmartPhoneApp.IFACIALMOCAPPC:
                    parse_pose_temp = parse_ifacialmocap_v1_pose(socket_string)
                    if parse_pose_temp[1] == True:
                        self.ifacialmocap_pose = parse_pose_temp[0]
                        self.poseIsPerfectsync = True
                    else:
                        self.ifacialmocap_pose = pose_cache
                        self.poseIsPerfectsync = isPerfectsync
                elif ifadd.SP_APP_MODE == SmartPhoneApp.VTUBESTUDIO:
                    parse_pose_temp = parse_vts_pose(socket_string)
                    if parse_pose_temp[2] == True:
                        self.ifacialmocap_pose = parse_pose_temp[0]
                        self.poseIsPerfectsync = True
                    else:
                        self.ifacialmocap_pose = pose_cache
                        self.poseIsPerfectsync = isPerfectsync
                    if self.vts_already_request:
                        self.vts_send_request()
                else:
                    parse_pose_temp = (create_default_ifacialmocap_pose(), False)
                    self.ifacialmocap_pose = parse_pose_temp[0]
                    self.poseIsPerfectsync = False
                self.show_status_indicator(True, parse_pose_temp[1])
            except:
                self.ifacialmocap_pose = pose_cache
                self.poseIsPerfectsync = isPerfectsync
                self.vmc_cache = []
                self.show_status_indicator(True, False)
        else:
            self.show_status_indicator(False, False)

        return self.ifacialmocap_pose

    def on_erase_background(self, event: wx.Event):
        pass

    def create_animation_panel(self, parent):
        self.animation_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.animation_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.animation_panel.SetSizer(self.animation_panel_sizer)
        self.animation_panel.SetAutoLayout(1)

        image_size = self.resized_image_size
        if True:
            self.input_panel = wx.Panel(self.animation_panel, size=(image_size, image_size + 142),
                                        style=wx.SIMPLE_BORDER)
            self.input_panel_sizer = wx.BoxSizer(wx.VERTICAL)
            self.input_panel.SetSizer(self.input_panel_sizer)
            self.input_panel.SetAutoLayout(1)
            self.animation_panel_sizer.Add(self.input_panel, 0, wx.FIXED_MINSIZE)

            self.source_image_panel = wx.Panel(self.input_panel, size=(image_size, image_size), style=wx.SIMPLE_BORDER)
            self.source_image_panel.Bind(wx.EVT_PAINT, self.paint_source_image_panel)
            self.source_image_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
            self.input_panel_sizer.Add(self.source_image_panel, 0, wx.FIXED_MINSIZE)

 #Listbox for select and switch images
            self.source_image_list = wx.ListBox(self.input_panel, size=(image_size, 100), style=wx.LB_NEEDED_SB)
            self.source_image_list.Bind(wx.EVT_LISTBOX, self.source_image_select)
            self.source_image_list.Bind(wx.EVT_LISTBOX_DCLICK, self.source_image_apply)
            self.source_image_list.Bind(wx.EVT_KEY_UP, self.source_image_press_enter)
            self.input_panel_sizer.Add(self.source_image_list, 1, wx.EXPAND)

            self.load_image_button = wx.Button(self.input_panel, wx.ID_ANY, "Load Image")
            self.input_panel_sizer.Add(self.load_image_button, 0, wx.EXPAND)
            self.load_image_button.Bind(wx.EVT_BUTTON, self.load_image)

            self.input_panel_sizer.Fit(self.input_panel)

        if True:
            self.pose_converter.init_pose_converter_panel(self.animation_panel)
            self.pose_converter.iris_left_slider.Enable(False)
            self.pose_converter.iris_right_slider.Enable(False)

            self.pose_converter.link_left_right_irises.Enable(False)

        if True:
            self.animation_left_panel = wx.Panel(self.animation_panel, style=wx.SIMPLE_BORDER)
            self.animation_left_panel_sizer = wx.BoxSizer(wx.VERTICAL)
            self.animation_left_panel.SetSizer(self.animation_left_panel_sizer)
            self.animation_left_panel.SetAutoLayout(1)
            self.animation_panel_sizer.Add(self.animation_left_panel, 0, wx.EXPAND)

            self.result_image_panel = wx.Panel(self.animation_left_panel, size=(image_size, image_size),
                                               style=wx.SIMPLE_BORDER)
            self.result_image_panel.Bind(wx.EVT_PAINT, self.paint_result_image_panel)
            self.result_image_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
            self.animation_left_panel_sizer.Add(self.result_image_panel, 0, wx.FIXED_MINSIZE)

            separator = wx.StaticLine(self.animation_left_panel, -1, size=(256, 5))
            self.animation_left_panel_sizer.Add(separator, 0, wx.EXPAND)

            background_text = wx.StaticText(self.animation_left_panel, label="--- Background ---",
                                            style=wx.ALIGN_CENTER)
            self.animation_left_panel_sizer.Add(background_text, 0, wx.EXPAND)

            self.output_background_choice = wx.Choice(
                self.animation_left_panel,
                choices=[
                    "TRANSPARENT",
                    "GREEN",
                    "BLUE",
                    "BLACK",
                    "WHITE"
                ])
            self.output_background_choice.SetSelection(0)
            self.output_background_choice.Bind(wx.EVT_CHOICE, self.background_changed)
            self.animation_left_panel_sizer.Add(self.output_background_choice, 0, wx.EXPAND)

            separator = wx.StaticLine(self.animation_left_panel, -1, size=(256, 5))
            self.animation_left_panel_sizer.Add(separator, 0, wx.EXPAND)

            self.fps_text = wx.StaticText(self.animation_left_panel, label="")
            self.animation_left_panel_sizer.Add(self.fps_text, wx.SizerFlags().Border())

            self.animation_left_panel_sizer.Fit(self.animation_left_panel)

        self.animation_panel_sizer.Fit(self.animation_panel)

    def create_ui(self):
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.main_sizer)
        self.SetAutoLayout(1)

        self.capture_pose_lock = threading.Lock()

        self.create_connection_panel(self)
        self.main_sizer.Add(self.connection_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

        self.create_animation_panel(self)
        self.main_sizer.Add(self.animation_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

#        self.create_capture_panel(self)
#        self.main_sizer.Add(self.capture_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

        self.main_sizer.Fit(self)

    def create_connection_panel(self, parent):
        self.connection_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.connection_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.connection_panel.SetSizer(self.connection_panel_sizer)
        self.connection_panel.SetAutoLayout(1)

#        Add reset button
        self.reset_button = wx.Button(self.connection_panel, label="CLEAR Images")
        self.connection_panel_sizer.Add(self.reset_button, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.reset_button.Bind(wx.EVT_BUTTON, self.reset_clicked)

        space_text = wx.StaticText(self.connection_panel, label="  ", style=wx.ALIGN_RIGHT)
        self.connection_panel_sizer.Add(space_text, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))

        self.reset_ok_button = wx.Button(self.connection_panel, label="  OK  ")
        self.connection_panel_sizer.Add(self.reset_ok_button, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.reset_ok_button.Bind(wx.EVT_BUTTON, self.reset_ok_clicked)

        self.reset_cancel_button = wx.Button(self.connection_panel, label="CANCEL")
        self.connection_panel_sizer.Add(self.reset_cancel_button, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.reset_cancel_button.Bind(wx.EVT_BUTTON, self.reset_cancel_clicked)

        reset_spacer_line = wx.StaticLine(self.connection_panel, style = wx.LI_VERTICAL, size = (5, 10))
        self.connection_panel_sizer.Add(reset_spacer_line, flag = wx.GROW | wx.RIGHT, border = 150)

        self.reset_ok_button.Disable()
        self.reset_cancel_button.Disable()


        capture_device_ip_text = wx.StaticText(self.connection_panel, label="Capture Device IP:", style=wx.ALIGN_RIGHT)
        self.connection_panel_sizer.Add(capture_device_ip_text, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))

        self.capture_device_ip_text_ctrl = wx.TextCtrl(self.connection_panel, value="192.168.0.1")
        self.connection_panel_sizer.Add(self.capture_device_ip_text_ctrl, wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

        self.start_capture_button = wx.Button(self.connection_panel, label="START CAPTURE!")
        self.connection_panel_sizer.Add(self.start_capture_button, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.start_capture_button.Bind(wx.EVT_BUTTON, self.on_start_capture)

        capture_status_text = wx.StaticText(self.connection_panel, label=" Status : ", style=wx.ALIGN_RIGHT)
        self.connection_panel_sizer.Add(capture_status_text, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))

        self.capture_status_indicator = wx.StaticText(self.connection_panel, label=" ● ", style=wx.ALIGN_RIGHT)
        self.connection_panel_sizer.Add(self.capture_status_indicator, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.show_status_indicator(False, False)
        
    def show_status_indicator(self, isReceive :bool = False, status :bool = False):
        if isReceive == True:
            if status == True:
                self.capture_status_indicator.SetForegroundColour("#00FF00")
                self.capture_status_indicator.SetBackgroundColour("#000000")
            else:
                self.capture_status_indicator.SetForegroundColour("#FFFF00")
                self.capture_status_indicator.SetBackgroundColour("#000000")
        else:
            self.capture_status_indicator.SetForegroundColour("#999999")
            self.capture_status_indicator.SetBackgroundColour("#000000")
        self.capture_status_indicator.Refresh()
        return

    def create_capture_panel(self, parent):
        self.capture_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.capture_panel_sizer = wx.FlexGridSizer(cols=5)
        for i in range(5):
            self.capture_panel_sizer.AddGrowableCol(i)
        self.capture_panel.SetSizer(self.capture_panel_sizer)
        self.capture_panel.SetAutoLayout(1)

        self.rotation_labels = {}
        self.rotation_value_labels = {}
        rotation_column_0 = self.create_rotation_column(self.capture_panel, RIGHT_EYE_BONE_ROTATIONS)
        self.capture_panel_sizer.Add(rotation_column_0, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))
        rotation_column_1 = self.create_rotation_column(self.capture_panel, LEFT_EYE_BONE_ROTATIONS)
        self.capture_panel_sizer.Add(rotation_column_1, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))
        rotation_column_2 = self.create_rotation_column(self.capture_panel, HEAD_BONE_ROTATIONS)
        self.capture_panel_sizer.Add(rotation_column_2, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))

    def create_rotation_column(self, parent, rotation_names):
        column_panel = wx.Panel(parent, style=wx.SIMPLE_BORDER)
        column_panel_sizer = wx.FlexGridSizer(cols=2)
        column_panel_sizer.AddGrowableCol(1)
        column_panel.SetSizer(column_panel_sizer)
        column_panel.SetAutoLayout(1)

        for rotation_name in rotation_names:
            self.rotation_labels[rotation_name] = wx.StaticText(
                column_panel, label=rotation_name, style=wx.ALIGN_RIGHT)
            column_panel_sizer.Add(self.rotation_labels[rotation_name],
                                   wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

            self.rotation_value_labels[rotation_name] = wx.TextCtrl(
                column_panel, style=wx.TE_RIGHT)
            self.rotation_value_labels[rotation_name].SetValue("0.00")
            self.rotation_value_labels[rotation_name].Disable()
            column_panel_sizer.Add(self.rotation_value_labels[rotation_name],
                                   wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

        column_panel.GetSizer().Fit(column_panel)
        return column_panel

    def paint_capture_panel(self, event: wx.Event):
        self.update_capture_panel(event)

    def update_capture_panel(self, event: wx.Event):
        data = self.ifacialmocap_pose
        for rotation_name in ROTATION_NAMES:
            value = data[rotation_name]
#            self.rotation_value_labels[rotation_name].SetValue("%0.2f" % value)

    @staticmethod
    def convert_to_100(x):
        return int(max(0.0, min(1.0, x)) * 100)

    def paint_source_image_panel(self, event: wx.Event):
        wx.BufferedPaintDC(self.source_image_panel, self.source_image_bitmap)

    def update_source_image_bitmap(self):
        dc = wx.MemoryDC()
        dc.SelectObject(self.source_image_bitmap)
        if self.wx_source_image is None:
            self.draw_nothing_yet_string(dc)
        else:
            draw_image = wx.Bitmap.ConvertToImage(self.wx_source_image)
            draw_image = draw_image.Scale(self.resized_image_size, self.resized_image_size, wx.IMAGE_QUALITY_BOX_AVERAGE)
            draw_wx_image = wx.Image.ConvertToBitmap(draw_image)
            dc.Clear()
            dc.DrawBitmap(draw_wx_image, 0, 0, True)
            if self.source_image_string is None:
                pass
            else:
                font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
                dc.SetFont(font)
                w, h = dc.GetTextExtent(self.source_image_string)
                dc.DrawText(self.source_image_string, (self.resized_image_size - w) // 2, h)

        del dc

    def draw_nothing_yet_string(self, dc):
        dc.Clear()
        font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
        dc.SetFont(font)
        w, h = dc.GetTextExtent("Nothing yet!")
        dc.DrawText("Nothing yet!", (self.resized_image_size - w) // 2, (self.resized_image_size - h) // 2)

    def paint_result_image_panel(self, event: wx.Event):
        wx.BufferedPaintDC(self.result_image_panel, self.result_image_bitmap)

    def pose_converter_prerendering_old(self, ifacialmocap_pose: Dict[str, float]):
        if ifadd.SP_APP_MODE == SmartPhoneApp.IPHONE:
            pose = self.convert_perfectsink(ifacialmocap_pose)
        elif ifadd.SP_APP_MODE == SmartPhoneApp.ANDROID:
            pose = self.convert_perfectsink(ifacialmocap_pose)
        elif ifadd.SP_APP_MODE == SmartPhoneApp.VMC:
            pose = self.convert_vmc_vrm0(ifacialmocap_pose)
        elif ifadd.SP_APP_MODE == SmartPhoneApp.VMCPERFECTSYNC:
            pose = self.convert_perfectsink(ifacialmocap_pose)
        elif ifadd.SP_APP_MODE == SmartPhoneApp.IFACIALMOCAPPC:
            pose = self.convert_perfectsink(ifacialmocap_pose)
        elif ifadd.SP_APP_MODE == SmartPhoneApp.VTUBESTUDIO:
            pose = self.convert_perfectsink(ifacialmocap_pose)
        else:
            pose = self.convert_vmc_vrm0(ifacialmocap_pose)
        
        return pose

    def pose_converter_prerendering(self, ifacialmocap_pose: Dict[str, float], isPerfectsync: bool):
        if isPerfectsync == True:
            pose = self.convert_perfectsink(ifacialmocap_pose)
        else:
            pose = self.convert_vmc_vrm0(ifacialmocap_pose)
        
        return pose

    def convert_perfectsink(self, ifacialmocap_pose: Dict[str, float]):
        pose = numpy.zeros((expression_layer_count, 3), dtype = numpy.int64)

        smile_value = \
            (ifacialmocap_pose[MOUTH_SMILE_LEFT] + ifacialmocap_pose[MOUTH_SMILE_RIGHT]) / 2.0 \
            + ifacialmocap_pose[MOUTH_SHRUG_UPPER]
        if smile_value < self.args.lower_smile_threshold:
            smile_degree = 0.0
        elif smile_value > self.args.upper_smile_threshold:
            smile_degree = 1.0
        else:
            smile_degree = (smile_value - self.args.lower_smile_threshold) / (
                    self.args.upper_smile_threshold - self.args.lower_smile_threshold)

        # Eyebrow
        if True:
            pose[PoseLayer.EyebrowMask.value][0] = ExpressionChoice.EyebrowMask

            brow_inner_up = ifacialmocap_pose[BROW_INNER_UP]
            brow_outer_up_right = ifacialmocap_pose[BROW_OUTER_UP_RIGHT]
            brow_outer_up_left = ifacialmocap_pose[BROW_OUTER_UP_LEFT]

            brow_up_left = clamp(brow_inner_up + brow_outer_up_left, 0.0, 1.0)
            brow_up_right = clamp(brow_inner_up + brow_outer_up_right, 0.0, 1.0)
            pose[PoseLayer.EyebrowLeft.value][2] = round(brow_up_left * 10.0)
            pose[PoseLayer.EyebrowRight.value][2] = round(brow_up_right * 10.0)

            brow_down_left = clamp(ifacialmocap_pose[BROW_DOWN_LEFT] / self.args.eyebrow_down_max_value, 0.0, 1.0)
            brow_down_right = clamp(ifacialmocap_pose[BROW_DOWN_RIGHT] / self.args.eyebrow_down_max_value, 0.0, 1.0)
            eyebrow_down_mode = self.pose_converter.args.eyebrow_down_mode
            if brow_down_left > smile_degree:
                if eyebrow_down_mode == EyebrowDownMode.TROUBLED:
                    pose[PoseLayer.EyebrowLeft.value][0] = ExpressionChoice.EyebrowTroubled
                elif eyebrow_down_mode == EyebrowDownMode.ANGRY:
                    pose[PoseLayer.EyebrowLeft.value][0] = ExpressionChoice.EyebrowAngry
                elif eyebrow_down_mode == EyebrowDownMode.LOWERED:
                    pose[PoseLayer.EyebrowLeft.value][0] = ExpressionChoice.EyebrowLowered
                # elif self.args.eyebrow_down_mode == EyebrowDownMode.SERIOUS:
                else:
                    pose[PoseLayer.EyebrowLeft.value][0] = ExpressionChoice.EyebrowSerious
                pose[PoseLayer.EyebrowLeft.value][1] = round((brow_down_left - smile_degree) * 10.0)
            else:
                brow_happy_value = smile_degree - brow_down_left
                pose[PoseLayer.EyebrowLeft.value][0] = ExpressionChoice.EyebrowHappy
                pose[PoseLayer.EyebrowLeft.value][1] = round((brow_happy_value) * 10.0)

            if brow_down_right > smile_degree:
                if eyebrow_down_mode == EyebrowDownMode.TROUBLED:
                    pose[PoseLayer.EyebrowRight.value][0] = ExpressionChoice.EyebrowTroubled
                elif eyebrow_down_mode == EyebrowDownMode.ANGRY:
                    pose[PoseLayer.EyebrowRight.value][0] = ExpressionChoice.EyebrowAngry
                elif eyebrow_down_mode == EyebrowDownMode.LOWERED:
                    pose[PoseLayer.EyebrowRight.value][0] = ExpressionChoice.EyebrowLowered
                # elif self.args.eyebrow_down_mode == EyebrowDownMode.SERIOUS:
                else:
                    pose[PoseLayer.EyebrowRight.value][0] = ExpressionChoice.EyebrowSerious
                pose[PoseLayer.EyebrowRight.value][1] = round((brow_down_left - smile_degree) * 10.0)
            else:
                brow_happy_value = smile_degree - brow_down_left
                pose[PoseLayer.EyebrowRight.value][0] = ExpressionChoice.EyebrowHappy
                pose[PoseLayer.EyebrowRight.value][1] = round((brow_happy_value) * 10.0)

        # Eye
        if True:
            wink_mode = self.pose_converter.args.wink_mode

            # Surprised
            if ifacialmocap_pose[EYE_WIDE_LEFT] >= self.args.eye_wide_max_value:
                pose[PoseLayer.EyeLeft.value][0] = ExpressionChoice.EyeSurprised
                pose[PoseLayer.EyelidLeft.value][0] = ExpressionChoice.EyelidSurprised
            else:
                pose[PoseLayer.EyeLeft.value][0] = ExpressionChoice.EyeNormal
            # Wink
                if wink_mode == WinkMode.NORMAL:
                    pose[PoseLayer.EyelidLeft.value][0] = ExpressionChoice.EyelidNormalHappy
                    pose[PoseLayer.EyelidLeft.value][1] = round(clamp(ifacialmocap_pose[EYE_BLINK_LEFT] / self.args.eye_blink_max_value, 0.0, 1.0) * 10.0)
                    pose[PoseLayer.EyelidLeft.value][2] = round(smile_degree * 10.0)
                else:
                    pose[PoseLayer.EyelidLeft.value][0] = ExpressionChoice.EyelidRelaxed
                    pose[PoseLayer.EyelidLeft.value][1] = round(clamp(ifacialmocap_pose[EYE_BLINK_LEFT] / self.args.eye_blink_max_value, 0.0, 1.0) * 10.0)

            # Lower eyelid
                cheek_squint_denom = self.args.cheek_squint_max_value - self.args.cheek_squint_min_value
                eye_raised_lower_left_index = clamp((ifacialmocap_pose[CHEEK_SQUINT_LEFT] - self.args.cheek_squint_min_value) / cheek_squint_denom, 0.0, 1.0)
                if pose[PoseLayer.EyelidLeft.value][1] < 1:
                    pose[PoseLayer.EyelidLeft.value][0] = ExpressionChoice.EyelidLower
                    pose[PoseLayer.EyelidLeft.value][1] = round(eye_raised_lower_left_index * 10.0)

            # Surprised
            if ifacialmocap_pose[EYE_WIDE_RIGHT] >= self.args.eye_wide_max_value:
                pose[PoseLayer.EyeRight.value][0] = ExpressionChoice.EyeSurprised
                pose[PoseLayer.EyelidRight.value][0] = ExpressionChoice.EyelidSurprised
            else:
                pose[PoseLayer.EyeRight.value][0] = ExpressionChoice.EyeNormal
            # Wink
                if wink_mode == WinkMode.NORMAL:
                    pose[PoseLayer.EyelidRight.value][0] = ExpressionChoice.EyelidNormalHappy
                    pose[PoseLayer.EyelidRight.value][1] = round(clamp(ifacialmocap_pose[EYE_BLINK_RIGHT] / self.args.eye_blink_max_value, 0.0, 1.0) * 10.0)
                    pose[PoseLayer.EyelidRight.value][2] = round(smile_degree * 10.0)
                else:
                    pose[PoseLayer.EyelidRight.value][0] = ExpressionChoice.EyelidRelaxed
                    pose[PoseLayer.EyelidRight.value][1] = round(clamp(ifacialmocap_pose[EYE_BLINK_RIGHT] / self.args.eye_blink_max_value, 0.0, 1.0) * 10.0)

            # Lower eyelid
                cheek_squint_denom = self.args.cheek_squint_max_value - self.args.cheek_squint_min_value
                eye_raised_lower_Right_index = clamp((ifacialmocap_pose[CHEEK_SQUINT_RIGHT] - self.args.cheek_squint_min_value) / cheek_squint_denom, 0.0, 1.0)
                if pose[PoseLayer.EyelidRight.value][1] < 1:
                    pose[PoseLayer.EyelidRight.value][0] = ExpressionChoice.EyelidLower
                    pose[PoseLayer.EyelidRight.value][1] = round(eye_raised_lower_Right_index * 10.0)

        # Iris rotation
        if True:
            eye_rotation_y = (ifacialmocap_pose[EYE_LOOK_IN_LEFT]
                              - ifacialmocap_pose[EYE_LOOK_OUT_LEFT]
                              - ifacialmocap_pose[EYE_LOOK_IN_RIGHT]
                              + ifacialmocap_pose[EYE_LOOK_OUT_RIGHT]) / 2.0 * self.args.eye_rotation_factor
            pose[PoseLayer.EyeLeft.value][2] = round(clamp(eye_rotation_y, -1.0, 1.0) * 5.0) + 5
            pose[PoseLayer.EyeRight.value][2] = round(clamp(eye_rotation_y, -1.0, 1.0) * 5.0) + 5

            eye_rotation_x = (ifacialmocap_pose[EYE_LOOK_UP_LEFT]
                              + ifacialmocap_pose[EYE_LOOK_UP_RIGHT]
                              - ifacialmocap_pose[EYE_LOOK_DOWN_LEFT]
                              - ifacialmocap_pose[EYE_LOOK_DOWN_RIGHT]) / 2.0 * self.args.eye_rotation_factor
            pose[PoseLayer.EyeLeft.value][1] = round(clamp(eye_rotation_x, -1.0, 1.0) * 5.0) + 5
            pose[PoseLayer.EyeRight.value][1] = round(clamp(eye_rotation_x, -1.0, 1.0) * 5.0) + 5

        # Iris size
        # if True:
        #     pose[self.iris_small_left_index] = self.args.iris_small_left
        #     pose[self.iris_small_right_index] = self.args.iris_small_right

        # Head rotation
        if True:
            # x_param = clamp(-ifacialmocap_pose[HEAD_BONE_X] * 180.0 / math.pi, -15.0, 15.0) / 15.0
            # pose[self.head_x_index] = x_param

            # y_param = clamp(-ifacialmocap_pose[HEAD_BONE_Y] * 180.0 / math.pi, -10.0, 10.0) / 10.0
            # pose[self.head_y_index] = y_param
            # pose[self.body_y_index] = y_param

            # z_param = clamp(ifacialmocap_pose[HEAD_BONE_Z] * 180.0 / math.pi, -15.0, 15.0) / 15.0
            # pose[self.neck_z_index] = z_param
            # pose[self.body_z_index] = z_param

            ifadd.POS_HEAD_X = ifacialmocap_pose[HEAD_BONE_X] * 180.0 / math.pi
            ifadd.POS_HEAD_Y = ifacialmocap_pose[HEAD_BONE_Y] * 180.0 / math.pi
            ifadd.POS_HEAD_Z = ifacialmocap_pose[HEAD_BONE_Z] * 180.0 / math.pi

            x_param = clamp(-(ifadd.POS_HEAD_X-ifadd.CAL_HEAD_X), -15.0, 15.0) / 15.0
            if x_param < 0:
                pose[PoseLayer.HeadX.value][0] = round(x_param * (-1024.0))
                pose[PoseLayer.HeadX.value][1] = 0
            else:
                pose[PoseLayer.HeadX.value][0] = round(x_param * (1024.0))
                pose[PoseLayer.HeadX.value][1] = 1

            y_param = clamp(-(ifadd.POS_HEAD_Y-ifadd.CAL_HEAD_Y), -10.0, 10.0) / 10.0
            if y_param < 0:
                pose[PoseLayer.HeadY.value][0] = round(y_param * (-1024.0))
                pose[PoseLayer.HeadY.value][1] = 0
            else:
                pose[PoseLayer.HeadY.value][0] = round(y_param * (1024.0))
                pose[PoseLayer.HeadY.value][1] = 1
            y_param_body = y_param * ifadd.CAL_BODY_Y / 10.0
            if y_param_body < 0:
                pose[PoseLayer.BodyY.value][0] = round(y_param_body * (-1024.0))
                pose[PoseLayer.BodyY.value][1] = 0
            else:
                pose[PoseLayer.BodyY.value][0] = round(y_param_body * (1024.0))
                pose[PoseLayer.BodyY.value][1] = 1

            z_param = clamp((ifadd.POS_HEAD_Z-ifadd.CAL_HEAD_Z), -15.0, 15.0) / 15.0
            if z_param < 0:
                pose[PoseLayer.BodyNeckZ.value][0] = round(z_param * (-1024.0))
                pose[PoseLayer.BodyNeckZ.value][1] = 0
            else:
                pose[PoseLayer.BodyNeckZ.value][0] = round(z_param * (1024.0))
                pose[PoseLayer.BodyNeckZ.value][1] = 1
            pose[PoseLayer.BodyNeckZ.value][2] = round(ifadd.CAL_BODY_Z / 2.0 + 5.0)

            # Mouth
        if True:
            jaw_open_denom = self.args.jaw_open_max_value - self.args.jaw_open_min_value
            mouth_open = clamp((ifacialmocap_pose[JAW_OPEN] - self.args.jaw_open_min_value) / jaw_open_denom, 0.0, 1.0)

            is_mouth_open = mouth_open > 0.0
            if not is_mouth_open:
                mouth_frown_value = clamp(
                    (ifacialmocap_pose[MOUTH_FROWN_LEFT] + ifacialmocap_pose[
                        MOUTH_FROWN_RIGHT]) / self.args.mouth_frown_max_value, 0.0, 1.0)
                if smile_degree > mouth_frown_value:
                    pose[PoseLayer.Mouth.value][0] = ExpressionChoice.BaseRaised
                    pose[PoseLayer.Mouth.value][1] = round(smile_degree * 10.0)
                else:
                    pose[PoseLayer.Mouth.value][0] = ExpressionChoice.BaseLowered
                    pose[PoseLayer.Mouth.value][1] = round(mouth_frown_value * 10.0)

            else:
                mouth_lower_down = clamp(
                    ifacialmocap_pose[MOUTH_LOWER_DOWN_LEFT] + ifacialmocap_pose[MOUTH_LOWER_DOWN_RIGHT], 0.0, 1.0)
                mouth_funnel = ifacialmocap_pose[MOUTH_FUNNEL]
                mouth_pucker = ifacialmocap_pose[MOUTH_PUCKER]

                mouth_point = [mouth_open, mouth_lower_down, mouth_funnel, mouth_pucker]

                aaa_point = [1.0, 1.0, 0.0, 0.0]
                iii_point = [0.0, 1.0, 0.0, 0.0]
                uuu_point = [0.5, 0.3, 0.25, 0.75]
                ooo_point = [1.0, 0.5, 0.5, 0.4]

                decomp = numpy.array([0, 0, 0, 0])
                M = numpy.array([
                    aaa_point,
                    iii_point,
                    uuu_point,
                    ooo_point
                ])

                def loss(decomp):
                    return numpy.linalg.norm(numpy.matmul(decomp, M) - mouth_point) \
                           + 0.01 * numpy.linalg.norm(decomp, ord=1)

                opt_result = scipy.optimize.minimize(
                    loss, decomp, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
                decomp = opt_result["x"]
                restricted_decomp = [decomp.item(0), decomp.item(1), decomp.item(2), decomp.item(3)]
                aaa_index = restricted_decomp[0]
                iii_index = restricted_decomp[1]
                mouth_funnel_denom = self.args.mouth_funnel_max_value - self.args.mouth_funnel_min_value
                ooo_alpha = clamp((mouth_funnel - self.args.mouth_funnel_min_value) / mouth_funnel_denom, 0.0, 1.0)
                uo_value = clamp(restricted_decomp[2] + restricted_decomp[3], 0.0, 1.0)
                uuu_index = uo_value * (1.0 - ooo_alpha)
                ooo_index = uo_value * ooo_alpha

                if aaa_index >= iii_index and aaa_index >= uuu_index and aaa_index >= ooo_index:
                    pose[PoseLayer.Mouth.value][0] = ExpressionChoice.BaseAAA
                    pose[PoseLayer.Mouth.value][1] = round(aaa_index * 10.0)
                elif iii_index >= uuu_index and iii_index >= ooo_index:
                    pose[PoseLayer.Mouth.value][0] = ExpressionChoice.BaseIII
                    pose[PoseLayer.Mouth.value][1] = round(iii_index * 10.0)
                elif uuu_index >= ooo_index:
                    pose[PoseLayer.Mouth.value][0] = ExpressionChoice.BaseUUU
                    pose[PoseLayer.Mouth.value][1] = round(uuu_index * 10.0)
                else:
                    pose[PoseLayer.Mouth.value][0] = ExpressionChoice.BaseOOO
                    pose[PoseLayer.Mouth.value][1] = round(ooo_index * 10.0)

        if self.pose_converter.panel is not None:
            frequency = self.pose_converter.breathing_frequency_slider.GetValue()
            if frequency == 0:
                value = 0.0
                pose[PoseLayer.Breath.value][0] = 0
                pose[PoseLayer.Breath.value][1] = 1
                self.pose_converter.breathing_start_time = time.time()
            else:
                period = 60.0 / frequency
                now = time.time()
                diff = now - self.pose_converter.breathing_start_time
                frac = (diff % period) / period
                value = (-math.cos(2 * math.pi * frac) + 1.0) / 2.0
                pose[PoseLayer.Breath.value][0] = round(value * 1024.0)
                pose[PoseLayer.Breath.value][1] = 1
            self.pose_converter.breathing_gauge.SetValue(int(1000 * value))

        return pose

    def convert_vmc_vrm0(self, ifacialmocap_pose: Dict[str, float]) -> List[float]:
        pose = numpy.zeros((expression_layer_count, 3), dtype = numpy.int64)

        smile_value = ifacialmocap_pose["Fun"]
        if smile_value < self.args.lower_smile_threshold:
            smile_degree = 0.0
        elif smile_value > self.args.upper_smile_threshold:
            smile_degree = 1.0
        else:
            smile_degree = (smile_value - self.args.lower_smile_threshold) / (
                    self.args.upper_smile_threshold - self.args.lower_smile_threshold)

        # Eyebrow
        if True:
            pose[PoseLayer.EyebrowMask.value][0] = ExpressionChoice.EyebrowMask

            brow_inner_up = ifacialmocap_pose["Surprised"] # ifacialmocap_pose[BROW_INNER_UP]
            brow_outer_up_right = 0.0 # ifacialmocap_pose[BROW_OUTER_UP_RIGHT]
            brow_outer_up_left = 0.0 # ifacialmocap_pose[BROW_OUTER_UP_LEFT]

            brow_up_left = clamp(brow_inner_up + brow_outer_up_left, 0.0, 1.0)
            brow_up_right = clamp(brow_inner_up + brow_outer_up_right, 0.0, 1.0)
            pose[PoseLayer.EyebrowLeft.value][2] = round(brow_up_left * 10.0)
            pose[PoseLayer.EyebrowRight.value][2] = round(brow_up_right * 10.0)

            # brow_down_left = (1.0 - smile_degree) \
            #                  * clamp(ifacialmocap_pose[BROW_DOWN_LEFT] / self.args.eyebrow_down_max_value, 0.0, 1.0)
            # brow_down_right = (1.0 - smile_degree) \
            #                   * clamp(ifacialmocap_pose[BROW_DOWN_RIGHT] / self.args.eyebrow_down_max_value, 0.0, 1.0)
            brow_down_left = 0.0
            brow_down_right = 0.0
            eyebrow_down_mode = self.pose_converter.args.eyebrow_down_mode
            if brow_down_left > smile_degree:
                if eyebrow_down_mode == EyebrowDownMode.TROUBLED:
                    pose[PoseLayer.EyebrowLeft.value][0] = ExpressionChoice.EyebrowTroubled
                elif eyebrow_down_mode == EyebrowDownMode.ANGRY:
                    pose[PoseLayer.EyebrowLeft.value][0] = ExpressionChoice.EyebrowAngry
                elif eyebrow_down_mode == EyebrowDownMode.LOWERED:
                    pose[PoseLayer.EyebrowLeft.value][0] = ExpressionChoice.EyebrowLowered
                # elif self.args.eyebrow_down_mode == EyebrowDownMode.SERIOUS:
                else:
                    pose[PoseLayer.EyebrowLeft.value][0] = ExpressionChoice.EyebrowSerious
                pose[PoseLayer.EyebrowLeft.value][1] = round((brow_down_left - smile_degree) * 10.0)
            else:
                brow_happy_value = smile_degree - brow_down_left
                pose[PoseLayer.EyebrowLeft.value][0] = ExpressionChoice.EyebrowHappy
                pose[PoseLayer.EyebrowLeft.value][1] = round((brow_happy_value) * 10.0)

            if brow_down_right > smile_degree:
                if eyebrow_down_mode == EyebrowDownMode.TROUBLED:
                    pose[PoseLayer.EyebrowRight.value][0] = ExpressionChoice.EyebrowTroubled
                elif eyebrow_down_mode == EyebrowDownMode.ANGRY:
                    pose[PoseLayer.EyebrowRight.value][0] = ExpressionChoice.EyebrowAngry
                elif eyebrow_down_mode == EyebrowDownMode.LOWERED:
                    pose[PoseLayer.EyebrowRight.value][0] = ExpressionChoice.EyebrowLowered
                # elif self.args.eyebrow_down_mode == EyebrowDownMode.SERIOUS:
                else:
                    pose[PoseLayer.EyebrowRight.value][0] = ExpressionChoice.EyebrowSerious
                pose[PoseLayer.EyebrowRight.value][1] = round((brow_down_left - smile_degree) * 10.0)
            else:
                brow_happy_value = smile_degree - brow_down_left
                pose[PoseLayer.EyebrowRight.value][0] = ExpressionChoice.EyebrowHappy
                pose[PoseLayer.EyebrowRight.value][1] = round((brow_happy_value) * 10.0)

        # Eye
        if True:
            wink_mode = self.pose_converter.args.wink_mode
            surprised_value = ifacialmocap_pose["Surprised"]
            eye_blink_left_value = ifacialmocap_pose["Blink_L"] + ifacialmocap_pose["Blink"]
            eye_blink_right_value = ifacialmocap_pose["Blink_R"] + ifacialmocap_pose["Blink"]

            # Surprised
            if (surprised_value - eye_blink_left_value) >= self.args.eye_wide_max_value:
                pose[PoseLayer.EyeLeft.value][0] = ExpressionChoice.EyeSurprised
                pose[PoseLayer.EyelidLeft.value][0] = ExpressionChoice.EyelidSurprised
            else:
                pose[PoseLayer.EyeLeft.value][0] = ExpressionChoice.EyeNormal
            # Wink
                if wink_mode == WinkMode.NORMAL:
                    pose[PoseLayer.EyelidLeft.value][0] = ExpressionChoice.EyelidNormalHappy
                    pose[PoseLayer.EyelidLeft.value][1] = round(clamp(eye_blink_left_value / self.args.eye_blink_max_value, 0.0, 1.0) * 10.0)
                    pose[PoseLayer.EyelidLeft.value][2] = round(smile_degree * 10.0)
                else:
                    pose[PoseLayer.EyelidLeft.value][0] = ExpressionChoice.EyelidRelaxed
                    pose[PoseLayer.EyelidLeft.value][1] = round(clamp(eye_blink_left_value / self.args.eye_blink_max_value, 0.0, 1.0) * 10.0)

            # Lower eyelid
                # # lower eyelid value is 0 
                # cheek_squint_denom = self.args.cheek_squint_max_value - self.args.cheek_squint_min_value
                # eye_raised_lower_left_index = clamp((ifacialmocap_pose[CHEEK_SQUINT_LEFT] - self.args.cheek_squint_min_value) / cheek_squint_denom, 0.0, 1.0)
                # if pose[PoseLayer.EyelidLeft.value][1] < 1:
                #     pose[PoseLayer.EyelidLeft.value][0] = ExpressionChoice.EyelidLower
                #     pose[PoseLayer.EyelidLeft.value][1] = round(eye_raised_lower_left_index * 10.0)

            # Surprised
            if (surprised_value - eye_blink_right_value) >= self.args.eye_wide_max_value:
                pose[PoseLayer.EyeRight.value][0] = ExpressionChoice.EyeSurprised
                pose[PoseLayer.EyelidRight.value][0] = ExpressionChoice.EyelidSurprised
            else:
                pose[PoseLayer.EyeRight.value][0] = ExpressionChoice.EyeNormal
            # Wink
                if wink_mode == WinkMode.NORMAL:
                    pose[PoseLayer.EyelidRight.value][0] = ExpressionChoice.EyelidNormalHappy
                    pose[PoseLayer.EyelidRight.value][1] = round(clamp(eye_blink_right_value / self.args.eye_blink_max_value, 0.0, 1.0) * 10.0)
                    pose[PoseLayer.EyelidRight.value][2] = round(smile_degree * 10.0)
                else:
                    pose[PoseLayer.EyelidRight.value][0] = ExpressionChoice.EyelidRelaxed
                    pose[PoseLayer.EyelidRight.value][1] = round(clamp(eye_blink_right_value / self.args.eye_blink_max_value, 0.0, 1.0) * 10.0)

            # Lower eyelid
                # # lower eyelid value is 0 
                # cheek_squint_denom = self.args.cheek_squint_max_value - self.args.cheek_squint_min_value
                # eye_raised_lower_Right_index = clamp((ifacialmocap_pose[CHEEK_SQUINT_RIGHT] - self.args.cheek_squint_min_value) / cheek_squint_denom, 0.0, 1.0)
                # if pose[PoseLayer.EyelidRight.value][1] < 1:
                #     pose[PoseLayer.EyelidRight.value][0] = ExpressionChoice.EyelidLower
                #     pose[PoseLayer.EyelidRight.value][1] = round(eye_raised_lower_Right_index * 10.0)

        # Iris rotation
        if True:
            eye_rotation_y = (ifacialmocap_pose[RIGHT_EYE_BONE_Y]
                              + ifacialmocap_pose[LEFT_EYE_BONE_Y]) / 2.0 * 1.0 * self.args.eye_rotation_factor
            pose[PoseLayer.EyeLeft.value][2] = round(clamp(eye_rotation_y, -1.0, 1.0) * 5.0) + 5
            pose[PoseLayer.EyeRight.value][2] = round(clamp(eye_rotation_y, -1.0, 1.0) * 5.0) + 5

            eye_rotation_x = (ifacialmocap_pose[RIGHT_EYE_BONE_X]
                              + ifacialmocap_pose[LEFT_EYE_BONE_X]) / 2.0 * 1.0 * self.args.eye_rotation_factor
            pose[PoseLayer.EyeLeft.value][1] = round(clamp(eye_rotation_x, -1.0, 1.0) * 5.0) + 5
            pose[PoseLayer.EyeRight.value][1] = round(clamp(eye_rotation_x, -1.0, 1.0) * 5.0) + 5

        # Iris size
        # if True:
        #     pose[self.iris_small_left_index] = self.args.iris_small_left
        #     pose[self.iris_small_right_index] = self.args.iris_small_right

        # Head rotation
        if True:
            # x_param = clamp(-ifacialmocap_pose[HEAD_BONE_X] * 180.0 / math.pi, -15.0, 15.0) / 15.0
            # pose[self.head_x_index] = x_param

            # y_param = clamp(-ifacialmocap_pose[HEAD_BONE_Y] * 180.0 / math.pi, -10.0, 10.0) / 10.0
            # pose[self.head_y_index] = y_param
            # pose[self.body_y_index] = y_param

            # z_param = clamp(ifacialmocap_pose[HEAD_BONE_Z] * 180.0 / math.pi, -15.0, 15.0) / 15.0
            # pose[self.neck_z_index] = z_param
            # pose[self.body_z_index] = z_param

            ifadd.POS_HEAD_X = ifacialmocap_pose[HEAD_BONE_X] * 180.0 / math.pi * 1.0
            ifadd.POS_HEAD_Y = ifacialmocap_pose[HEAD_BONE_Y] * 180.0 / math.pi * 1.0
            ifadd.POS_HEAD_Z = ifacialmocap_pose[HEAD_BONE_Z] * 180.0 / math.pi * 1.0

            x_param = clamp(-(ifadd.POS_HEAD_X-ifadd.CAL_HEAD_X), -15.0, 15.0) / 15.0
            if x_param < 0:
                pose[PoseLayer.HeadX.value][0] = round(x_param * (-1024.0))
                pose[PoseLayer.HeadX.value][1] = 0
            else:
                pose[PoseLayer.HeadX.value][0] = round(x_param * (1024.0))
                pose[PoseLayer.HeadX.value][1] = 1

            y_param = clamp(-(ifadd.POS_HEAD_Y-ifadd.CAL_HEAD_Y), -10.0, 10.0) / 10.0
            if y_param < 0:
                pose[PoseLayer.HeadY.value][0] = round(y_param * (-1024.0))
                pose[PoseLayer.HeadY.value][1] = 0
            else:
                pose[PoseLayer.HeadY.value][0] = round(y_param * (1024.0))
                pose[PoseLayer.HeadY.value][1] = 1
            y_param_body = y_param * ifadd.CAL_BODY_Y / 10.0
            if y_param_body < 0:
                pose[PoseLayer.BodyY.value][0] = round(y_param_body * (-1024.0))
                pose[PoseLayer.BodyY.value][1] = 0
            else:
                pose[PoseLayer.BodyY.value][0] = round(y_param_body * (1024.0))
                pose[PoseLayer.BodyY.value][1] = 1

            z_param = clamp((ifadd.POS_HEAD_Z-ifadd.CAL_HEAD_Z), -15.0, 15.0) / 15.0
            if z_param < 0:
                pose[PoseLayer.BodyNeckZ.value][0] = round(z_param * (-1024.0))
                pose[PoseLayer.BodyNeckZ.value][1] = 0
            else:
                pose[PoseLayer.BodyNeckZ.value][0] = round(z_param * (1024.0))
                pose[PoseLayer.BodyNeckZ.value][1] = 1
            pose[PoseLayer.BodyNeckZ.value][2] = round(ifadd.CAL_BODY_Z / 2.0 + 5.0)

            # Mouth
        if True:
            jaw_open_denom = self.args.jaw_open_max_value - self.args.jaw_open_min_value
            mouth_sum_aiueo = ifacialmocap_pose["A"] + ifacialmocap_pose["I"] + ifacialmocap_pose["U"] + ifacialmocap_pose["E"] + ifacialmocap_pose["O"]
            mouth_open = clamp((mouth_sum_aiueo - self.args.jaw_open_min_value) / jaw_open_denom, 0.0, 1.0)

            is_mouth_open = mouth_open > 0.0
            if not is_mouth_open:
                # mouth_frown_value = clamp(
                #     (ifacialmocap_pose[MOUTH_FROWN_LEFT] + ifacialmocap_pose[
                #         MOUTH_FROWN_RIGHT]) / self.args.mouth_frown_max_value, 0.0, 1.0)
                if True :
                    pose[PoseLayer.Mouth.value][0] = ExpressionChoice.BaseRaised
                    pose[PoseLayer.Mouth.value][1] = round(smile_degree * 10.0)
                # else:
                #     pose[PoseLayer.Mouth.value][0] = ExpressionChoice.BaseLowered
                #     pose[PoseLayer.Mouth.value][1] = round(mouth_frown_value * 10.0)

            else:
                aaa_index = clamp(ifacialmocap_pose["A"], 0.0, 1.0)
                iii_index = clamp(ifacialmocap_pose["I"], 0.0, 1.0)
                uuu_index = clamp(ifacialmocap_pose["U"], 0.0, 1.0)
                eee_index = clamp(ifacialmocap_pose["E"], 0.0, 1.0)
                ooo_index = clamp(ifacialmocap_pose["O"], 0.0, 1.0)

                if aaa_index >= iii_index and aaa_index >= uuu_index and aaa_index >= eee_index and aaa_index >= ooo_index:
                    pose[PoseLayer.Mouth.value][0] = ExpressionChoice.BaseAAA
                    pose[PoseLayer.Mouth.value][1] = round(aaa_index * 10.0)
                elif eee_index >= iii_index and eee_index >= uuu_index and eee_index >= ooo_index:
                    pose[PoseLayer.Mouth.value][0] = ExpressionChoice.BaseEEE
                    pose[PoseLayer.Mouth.value][1] = round(eee_index * 10.0)
                elif iii_index >= uuu_index and iii_index >= ooo_index:
                    pose[PoseLayer.Mouth.value][0] = ExpressionChoice.BaseIII
                    pose[PoseLayer.Mouth.value][1] = round(iii_index * 10.0)
                elif uuu_index >= ooo_index:
                    pose[PoseLayer.Mouth.value][0] = ExpressionChoice.BaseUUU
                    pose[PoseLayer.Mouth.value][1] = round(uuu_index * 10.0)
                else:
                    pose[PoseLayer.Mouth.value][0] = ExpressionChoice.BaseOOO
                    pose[PoseLayer.Mouth.value][1] = round(ooo_index * 10.0)

        if self.pose_converter.panel is not None:
            frequency = self.pose_converter.breathing_frequency_slider.GetValue()
            if frequency == 0:
                value = 0.0
                pose[PoseLayer.Breath.value][0] = 0
                pose[PoseLayer.Breath.value][1] = 1
                self.pose_converter.breathing_start_time = time.time()
            else:
                period = 60.0 / frequency
                now = time.time()
                diff = now - self.pose_converter.breathing_start_time
                frac = (diff % period) / period
                value = (-math.cos(2 * math.pi * frac) + 1.0) / 2.0
                pose[PoseLayer.Breath.value][0] = round(value * 1024.0)
                pose[PoseLayer.Breath.value][1] = 1
            self.pose_converter.breathing_gauge.SetValue(int(1000 * value))

        return pose

    def update_result_image_bitmap(self, event: Optional[wx.Event] = None):
        ifacialmocap_pose = self.read_ifacialmocap_pose()
        # print(ifacialmocap_pose)
        # ifacialmocap_pose = create_default_ifacialmocap_pose()
        current_pose = self.pose_converter_prerendering(ifacialmocap_pose, self.poseIsPerfectsync)
        # print(current_pose)
        self.last_pose = current_pose

        if self.np_source_large is None:
            # dc = wx.MemoryDC()
            # dc.SelectObject(self.result_image_bitmap)
            # self.draw_nothing_yet_string(dc)
            # del dc

            background = numpy.zeros((self.image_view_size, self.image_view_size, 4), dtype = numpy.uint8)
            background_choice = self.output_background_choice.GetSelection()
            if background_choice == 0:
                pass
            else:
                background[ :, :, 3] = 255
                if background_choice == 1:
                    background[ :, :, 1] = 255
                elif background_choice == 2:
                    background[ :, :, 2] = 255
                elif background_choice == 3:
                    pass
                else:
                    background[ :, :, 0:3] = 255
            output_image = background
            h, w ,c = output_image.shape
            numpy_image = output_image
            # print(numpy_image[0,0])
            # wx_image = wx.ImageFromBuffer(numpy_image.shape[0],
            #                               numpy_image.shape[1],
            #                               numpy_image[:, :, 0:3].tobytes(),
            #                               numpy_image[:, :, 3].tobytes())
            # wx_bitmap = wx_image.ConvertToBitmap()
            wx_bitmap = wx.Bitmap.FromBufferRGBA(self.image_size, self.image_size, numpy_image)

            dc = wx.MemoryDC()
            dc.SelectObject(self.result_image_bitmap)
            dc.Clear()
            dc.DrawBitmap(wx_bitmap,
                          (self.image_view_size - numpy_image.shape[0]) // 2,
                          (self.image_view_size - numpy_image.shape[1]) // 2, True)
            del dc
            # self.Refresh()
            self.result_image_panel.Refresh()
            return

        comp_image = comp_expression_image(current_pose, self.prerendering_numpy_image)
        map_image = comp_map_image(current_pose, self.prerendering_numpy_image)
        output_image = transform_image_with_map(comp_image, map_image)

        
        numpy_image = output_image.astype(numpy.uint8)

        background_choice = self.output_background_choice.GetSelection()
        if background_choice == 0:
            pass
        else:
            background = numpy.zeros((self.image_view_size, self.image_view_size, 4), dtype = numpy.uint8)
            background[ :, :, 3] = 255
            if background_choice == 1:
                background[ :, :, 1] = 255
            elif background_choice == 2:
                background[ :, :, 2] = 255
            elif background_choice == 3:
                pass
            else:
                background[ :, :, 0:3] = 255
            numpy_image = blend_numpy_image(numpy_image, background)

        # wx_image = wx.ImageFromBuffer(numpy_image.shape[0],
        #                               numpy_image.shape[1],
        #                               numpy_image[:, :, 0:3].tobytes(),
        #                               numpy_image[:, :, 3].tobytes())
        # wx_bitmap = wx_image.ConvertToBitmap()
        wx_bitmap = wx.Bitmap.FromBufferRGBA(self.image_size, self.image_size, numpy_image)
        dc = wx.MemoryDC()
        dc.SelectObject(self.result_image_bitmap)
        dc.Clear()
        dc.DrawBitmap(wx_bitmap,
                      (self.image_view_size - numpy_image.shape[0]) // 2,
                      (self.image_view_size - numpy_image.shape[1]) // 2, True)
        del dc

        time_now = time.time_ns()
        if self.last_update_time is not None:
            elapsed_time = time_now - self.last_update_time
            fps = 1.0 / (elapsed_time / 10**9)
            self.fps_statistics.add_fps(fps)
            self.fps_text.SetLabelText("FPS = %0.2f" % self.fps_statistics.get_average_fps())
        self.last_update_time = time_now

        self.result_image_panel.Refresh()

    def change_composition_algorithm(self, event: wx.Event):
        self.algorithm_mode = self.algorithm_mode_choice.GetSelection()
        self.create_divided_images()
        pass

    def extract_numpy_image_from_PIL_image_raw(pil_image, has_alpha=True, scale=2.0, offset=-1.0):
        if has_alpha:
            num_channel = 4
        else:
            num_channel = 3
        image_size = pil_image.width

        # search for transparent pixels(alpha==0) and change them to [0 0 0 0] to avoid the color influence to the model
        for i, px in enumerate(pil_image.getdata()):
            if px[3] <= 0:
                y = i // image_size
                x = i % image_size
                pil_image.putpixel((x, y), (0, 0, 0, 0))

        raw_image = numpy.asarray(pil_image)
        image = (raw_image / 255.0).reshape(image_size, image_size, num_channel)
        image[:, :, 0:3] = numpy_srgb_to_linear(image[:, :, 0:3])
        image = image \
                    .reshape(image_size * image_size, num_channel) \
                    .transpose() \
                    .reshape(num_channel, image_size, image_size) * scale + offset
        return image

    def load_image(self, event: wx.Event):
        dir_name = "data/images"
        file_dialog = wx.FileDialog(self, "Choose an image", dir_name, "", "*.png", wx.FD_OPEN)
        if file_dialog.ShowModal() == wx.ID_OK:
            image_name = file_dialog.GetFilename()
            image_file_name = os.path.join(file_dialog.GetDirectory(), image_name)
            try:
            # if True:
                pil_image = resize_PIL_image(
                    extract_PIL_image_from_filelike(image_file_name),
                    (self.image_size, self.image_size))
                w, h = pil_image.size
                if pil_image.mode != 'RGBA':
                    # self.source_image_string = "Image must have alpha channel!"
                    self.source_image_string = None
                    tip = wx.adv.RichToolTip("Notice", "Loading Error.\nImage must have alpha channel!")
                    tip.SetTimeout(10000, 0)
                    tip.ShowFor(self.load_image_button)
                else:
                    self.source_image_string = None
                    image_list_index = self.source_image_list.GetCount()
                    wx_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                    np_image_large = numpy.asarray(pil_image)

                    image_filepath_base = file_dialog.GetDirectory()
                    image_filename_base = os.path.splitext(image_name)[0]
                    image_loadpath = os.path.join(image_filepath_base, image_filename_base + "_tha")
                    if os.path.isdir(image_loadpath) == False:
                        tip = wx.adv.RichToolTip("Notice", "Loading Error.\nPrerendering Images Not Found!")
                        tip.SetTimeout(10000, 0)
                        tip.ShowFor(self.load_image_button)
                        raise RuntimeError("Prerendering Images Not Found!")
                    prerendering_image, image_missing = load_prerendering_images(image_loadpath, image_filename_base)
                    if image_missing == True:
                        image_name = image_name + "  (Some Prerendering Images Are Missing!)"
                    self.source_image_list.Append(image_name, [pil_image, wx_image, np_image_large, prerendering_image, image_file_name])
                    self.source_image_list.SetSelection(image_list_index)
                    self.wx_source_image = wx_image
                    if image_list_index == 0:
                        tip = wx.adv.RichToolTip("Click to preview image.\nDoubleClick to output animation.", "") # "Quick Guide"
                        tip.SetTimeout(10000, 0)
                        tip.ShowFor(self.source_image_list)
                self.update_source_image_bitmap()
            except Exception as e:
                print(e)
            #     message_dialog = wx.MessageDialog(self, "Could not load image " + image_file_name, "Poser", wx.OK)
            #     message_dialog.ShowModal()
            #     message_dialog.Destroy()
        file_dialog.Destroy()
        self.Refresh()

    def source_image_select(self, event: wx.Event):
        obj = event.GetEventObject()
        select_index = obj.GetSelection()
        self.last_show_index = select_index
        image_sets = obj.GetClientData(select_index)
        self.source_image_string = None
        self.wx_source_image = image_sets[1]
        self.update_source_image_bitmap()
        self.Refresh()

    def source_image_apply(self, event: wx.Event):
        obj = event.GetEventObject()
        select_index = obj.GetSelection()
        self.last_output_index = select_index
        image_sets = obj.GetClientData(select_index)
        self.source_image_string = None
        self.wx_source_image = image_sets[1]
        self.update_source_image_bitmap()
        self.prerendering_numpy_image = image_sets[3]
        self.np_source_large = image_sets[2]
        self.last_pose = None
        self.update_result_image_bitmap()
        self.Refresh()

    def source_image_press_enter(self, event: wx.Event):
        Code_Enter = 13
        if event.GetKeyCode() == Code_Enter:
            # source_image_apply
            obj = event.GetEventObject()
            select_index = obj.GetSelection()
            self.last_output_index = select_index
            image_sets = obj.GetClientData(select_index)
            self.source_image_string = None
            self.wx_source_image = image_sets[1]
            self.update_source_image_bitmap()
            self.prerendering_numpy_image = image_sets[3]
            self.np_source_large = image_sets[2]
            self.last_pose = None
            self.update_result_image_bitmap()
            self.Refresh()
        else:
            event.Skip()

    def reset_clicked(self, event: wx.Event):
        self.reset_button.Disable()
        self.reset_ok_button.Enable()
        self.reset_cancel_button.Enable()
        self.reset_cancel_button.SetFocus()

    def reset_ok_clicked(self, event: wx.Event):
        self.reset_ok_button.Disable()
        self.reset_cancel_button.Disable()
        self.reset_button.Enable()

        self.source_image_list.Clear()

        self.ifacialmocap_pose = create_default_ifacialmocap_pose()
        self.wx_source_image = None
        self.np_source_large = None
        self.last_pose = None
        self.prerendering_numpy_image = None

        self.update_source_image_bitmap()
        self.update_result_image_bitmap()
        self.Refresh()

        self.reset_button.SetFocus()

    def reset_cancel_clicked(self, event: wx.Event):
        self.reset_ok_button.Disable()
        self.reset_cancel_button.Disable()
        self.reset_button.Enable()
        self.reset_button.SetFocus()

    def background_changed(self, event: wx.Event):
        self.same_pose_count = 0

if __name__ == "__main__":
    if os.path.exists('tha3sw_config.json'):
        try:
            with open('tha3sw_config.json') as f:
                    config_json_dict = json.load(f)
        except Exception as e:# json.JSONDecodeError:
            print(e)
            config_json_dict = {}
    else:
        print("JSON file is not found!")
        config_json_dict = {}

    save_config = config_json_dict.get("save_config", None)
    if save_config == True:
        timer_preserve = int(config_json_dict.get("timer", 20))
        port_preserve = int(config_json_dict.get("port", 49983))
    else:
        timer_preserve = 20
        port_preserve = 49983

    parser = argparse.ArgumentParser(description='Control characters with movement captured by iFacialMocap.')
    parser.add_argument(
        '--timer',
        type=int,
        required=False,
        default=timer_preserve,
        # choices=range(5, 2000),
        help='Animation cycle ; 5-2000[ms].')
    parser.add_argument(
        '--port',
        type=int,
        required=False,
        default=port_preserve,
        # choices=range(0, 65535),
        help='Network port number to recieve motioncapture ; default is 49983.')
    args = parser.parse_args()

    from tha3.mocap.ifacialmocap_poser_converter_25 import create_ifacialmocap_pose_converter

    pose_converter = create_ifacialmocap_pose_converter()

    np_temp = numpy.zeros((512, 512, 2), dtype = numpy.float64)
    image_temp = numpy.zeros((512, 512, 4), dtype = numpy.uint8)
    blend_temp = blend_numpy_image(image_temp, image_temp)
    trans_temp = transform_image_with_map(image_temp, np_temp, 512)

    port_number = args.port
    if port_number < 0:
        print("--port value is too small !!!")
        port_number = 49983
    elif port_number > 65535:
        print("--port value is too large !!!")
        port_number = 49983
    print(f"Receive motioncapture from port {port_number} .")

    app = wx.App()
    main_frame = MainFrame(pose_converter, port_number)
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

    if save_config == True:
        try:
            config_json_dict["timer"] = maintimer
            config_json_dict["port"] = port_number
            with open('tha3sw_config.json', 'w') as f:
                json.dump(config_json_dict, f, indent=4)
        except Exception as e:
            print (e)

    main_frame.animation_timer.Start(maintimer)
    app.MainLoop()
