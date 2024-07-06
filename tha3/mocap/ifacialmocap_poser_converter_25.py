import math
import time
from enum import Enum
from typing import Optional, Dict, List

import numpy
import scipy.optimize
import wx

import tha3.mocap.ifacialmocap_add as ifadd
from tha3.mocap.ifacialmocap_constants import RIGHT_EYE_BONE_X, RIGHT_EYE_BONE_Y, RIGHT_EYE_BONE_Z, LEFT_EYE_BONE_X, LEFT_EYE_BONE_Y, LEFT_EYE_BONE_Z, \
    HEAD_BONE_QUAT, LEFT_EYE_BONE_QUAT, RIGHT_EYE_BONE_QUAT

from tha3.mocap.ifacialmocap_constants import MOUTH_SMILE_LEFT, MOUTH_SHRUG_UPPER, MOUTH_SMILE_RIGHT, \
    BROW_INNER_UP, BROW_OUTER_UP_RIGHT, BROW_OUTER_UP_LEFT, BROW_DOWN_LEFT, BROW_DOWN_RIGHT, EYE_WIDE_LEFT, \
    EYE_WIDE_RIGHT, EYE_BLINK_LEFT, EYE_BLINK_RIGHT, CHEEK_SQUINT_LEFT, CHEEK_SQUINT_RIGHT, EYE_LOOK_IN_LEFT, \
    EYE_LOOK_OUT_LEFT, EYE_LOOK_IN_RIGHT, EYE_LOOK_OUT_RIGHT, EYE_LOOK_UP_LEFT, EYE_LOOK_UP_RIGHT, EYE_LOOK_DOWN_RIGHT, \
    EYE_LOOK_DOWN_LEFT, HEAD_BONE_X, HEAD_BONE_Y, HEAD_BONE_Z, JAW_OPEN, MOUTH_FROWN_LEFT, MOUTH_FROWN_RIGHT, \
    MOUTH_LOWER_DOWN_LEFT, MOUTH_LOWER_DOWN_RIGHT, MOUTH_FUNNEL, MOUTH_PUCKER
from tha3.mocap.ifacialmocap_pose_converter import IFacialMocapPoseConverter
from tha3.poser.modes.pose_parameters import get_pose_parameters

import tha3.mocap.ifacialmocap_add as ifadd

class SmartPhoneApp(Enum):
    IPHONE = 1
    ANDROID = 2
    VMC = 3
    VMCPERFECTSYNC = 4
    IFACIALMOCAPPC = 5
    VTUBESTUDIO = 6

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


class IFacialMocapPoseConverter25Args:
    def __init__(self,
                 lower_smile_threshold: float = 0.4,
                 upper_smile_threshold: float = 0.6,
                 eyebrow_down_mode: EyebrowDownMode = EyebrowDownMode.ANGRY,
                 wink_mode: WinkMode = WinkMode.NORMAL,
                 eye_surprised_max_value: float = 0.5,
                 eye_wink_max_value: float = 0.8,
                 eyebrow_down_max_value: float = 0.4,
                 cheek_squint_min_value: float = 0.1,
                 cheek_squint_max_value: float = 0.7,
                 eye_rotation_factor: float = 1.0 / 0.75,
                 jaw_open_min_value: float = 0.1,
                 jaw_open_max_value: float = 0.4,
                 mouth_frown_max_value: float = 0.6,
                 mouth_funnel_min_value: float = 0.25,
                 mouth_funnel_max_value: float = 0.5,
                 iris_small_left=0.0,
                 iris_small_right=0.0):
#                 sp_app_choice: SmartPhoneApp = SmartPhoneApp.IPHONE
        self.iris_small_right = iris_small_left
        self.iris_small_left = iris_small_right
        self.wink_mode = wink_mode
        self.mouth_funnel_max_value = mouth_funnel_max_value
        self.mouth_funnel_min_value = mouth_funnel_min_value
        self.mouth_frown_max_value = mouth_frown_max_value
        self.jaw_open_max_value = jaw_open_max_value
        self.jaw_open_min_value = jaw_open_min_value
        self.eye_rotation_factor = eye_rotation_factor
        self.cheek_squint_max_value = cheek_squint_max_value
        self.cheek_squint_min_value = cheek_squint_min_value
        self.eyebrow_down_max_value = eyebrow_down_max_value
        self.eye_blink_max_value = eye_wink_max_value
        self.eye_wide_max_value = eye_surprised_max_value
        self.eyebrow_down_mode = eyebrow_down_mode
        self.lower_smile_threshold = lower_smile_threshold
        self.upper_smile_threshold = upper_smile_threshold
#        self.sp_app_choice = sp_app_choice


class IFacialMocapPoseConverter25(IFacialMocapPoseConverter):
    def __init__(self, args: Optional[IFacialMocapPoseConverter25Args] = None):
        super().__init__()
        if args is None:
            args = IFacialMocapPoseConverter25Args()
        self.args = args
        pose_parameters = get_pose_parameters()
        self.pose_size = 45

        self.eyebrow_troubled_left_index = pose_parameters.get_parameter_index("eyebrow_troubled_left")
        self.eyebrow_troubled_right_index = pose_parameters.get_parameter_index("eyebrow_troubled_right")
        self.eyebrow_angry_left_index = pose_parameters.get_parameter_index("eyebrow_angry_left")
        self.eyebrow_angry_right_index = pose_parameters.get_parameter_index("eyebrow_angry_right")
        self.eyebrow_happy_left_index = pose_parameters.get_parameter_index("eyebrow_happy_left")
        self.eyebrow_happy_right_index = pose_parameters.get_parameter_index("eyebrow_happy_right")
        self.eyebrow_raised_left_index = pose_parameters.get_parameter_index("eyebrow_raised_left")
        self.eyebrow_raised_right_index = pose_parameters.get_parameter_index("eyebrow_raised_right")
        self.eyebrow_lowered_left_index = pose_parameters.get_parameter_index("eyebrow_lowered_left")
        self.eyebrow_lowered_right_index = pose_parameters.get_parameter_index("eyebrow_lowered_right")
        self.eyebrow_serious_left_index = pose_parameters.get_parameter_index("eyebrow_serious_left")
        self.eyebrow_serious_right_index = pose_parameters.get_parameter_index("eyebrow_serious_right")

        self.eye_surprised_left_index = pose_parameters.get_parameter_index("eye_surprised_left")
        self.eye_surprised_right_index = pose_parameters.get_parameter_index("eye_surprised_right")
        self.eye_wink_left_index = pose_parameters.get_parameter_index("eye_wink_left")
        self.eye_wink_right_index = pose_parameters.get_parameter_index("eye_wink_right")
        self.eye_happy_wink_left_index = pose_parameters.get_parameter_index("eye_happy_wink_left")
        self.eye_happy_wink_right_index = pose_parameters.get_parameter_index("eye_happy_wink_right")
        self.eye_relaxed_left_index = pose_parameters.get_parameter_index("eye_relaxed_left")
        self.eye_relaxed_right_index = pose_parameters.get_parameter_index("eye_relaxed_right")
        self.eye_raised_lower_eyelid_left_index = pose_parameters.get_parameter_index("eye_raised_lower_eyelid_left")
        self.eye_raised_lower_eyelid_right_index = pose_parameters.get_parameter_index("eye_raised_lower_eyelid_right")

        self.iris_small_left_index = pose_parameters.get_parameter_index("iris_small_left")
        self.iris_small_right_index = pose_parameters.get_parameter_index("iris_small_right")

        self.iris_rotation_x_index = pose_parameters.get_parameter_index("iris_rotation_x")
        self.iris_rotation_y_index = pose_parameters.get_parameter_index("iris_rotation_y")

        self.head_x_index = pose_parameters.get_parameter_index("head_x")
        self.head_y_index = pose_parameters.get_parameter_index("head_y")
        self.neck_z_index = pose_parameters.get_parameter_index("neck_z")

        self.mouth_aaa_index = pose_parameters.get_parameter_index("mouth_aaa")
        self.mouth_iii_index = pose_parameters.get_parameter_index("mouth_iii")
        self.mouth_uuu_index = pose_parameters.get_parameter_index("mouth_uuu")
        self.mouth_eee_index = pose_parameters.get_parameter_index("mouth_eee")
        self.mouth_ooo_index = pose_parameters.get_parameter_index("mouth_ooo")

        self.mouth_lowered_corner_left_index = pose_parameters.get_parameter_index("mouth_lowered_corner_left")
        self.mouth_lowered_corner_right_index = pose_parameters.get_parameter_index("mouth_lowered_corner_right")
        self.mouth_raised_corner_left_index = pose_parameters.get_parameter_index("mouth_raised_corner_left")
        self.mouth_raised_corner_right_index = pose_parameters.get_parameter_index("mouth_raised_corner_right")

        self.body_y_index = pose_parameters.get_parameter_index("body_y")
        self.body_z_index = pose_parameters.get_parameter_index("body_z")
        self.breathing_index = pose_parameters.get_parameter_index("breathing")

        self.breathing_start_time = time.time()

        self.panel = None

    def init_pose_converter_panel(self, parent):
        self.panel = wx.Panel(parent, style=wx.SIMPLE_BORDER)
        self.panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel.SetSizer(self.panel_sizer)
        self.panel.SetAutoLayout(1)
        parent.GetSizer().Add(self.panel, 0, wx.EXPAND)

        if True:
            sp_app_text = wx.StaticText(self.panel, label=" --- Motion Capture Method--- ",
                                                   style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(sp_app_text, 0, wx.EXPAND)

            self.sp_app_choice = wx.Choice(
                self.panel,
                choices=[
                    "iFacialMocap (iPhone)",
                    "MeowFace (Android)",
                    "VMC Protocol (VRM0/VRM1)",
                    "VMC Protocol (Perfect Sync)",
                    "iFacialMocap (PC)",
                    "VTubeStudio (iPhone/Android)",
                ])
            self.sp_app_choice.SetSelection(2)
            self.panel_sizer.Add(self.sp_app_choice, 0, wx.EXPAND)
            self.sp_app_choice.Bind(wx.EVT_CHOICE, self.change_sp_app)
            self.change_sp_app(self)

            separator = wx.StaticLine(self.panel, -1, size=(236, 3))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

        if True:
            eyebrow_down_mode_text = wx.StaticText(self.panel, label=" --- Eyebrow Down Mode --- ",
                                                   style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(eyebrow_down_mode_text, 0, wx.EXPAND)

            self.eyebrow_down_mode_choice = wx.Choice(
                self.panel,
                choices=[
                    "ANGRY",
                    "TROUBLED",
                    "SERIOUS",
                    "LOWERED",
                ])
            self.eyebrow_down_mode_choice.SetSelection(0)
            self.panel_sizer.Add(self.eyebrow_down_mode_choice, 0, wx.EXPAND)
            self.eyebrow_down_mode_choice.Bind(wx.EVT_CHOICE, self.change_eyebrow_down_mode)

            separator = wx.StaticLine(self.panel, -1, size=(234, 3))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

        if True:
            wink_mode_text = wx.StaticText(self.panel, label=" --- Wink Mode --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(wink_mode_text, 0, wx.EXPAND)

            self.wink_mode_choice = wx.Choice(
                self.panel,
                choices=[
                    "NORMAL",
                    "RELAXED",
                ])
            self.wink_mode_choice.SetSelection(0)
            self.panel_sizer.Add(self.wink_mode_choice, 0, wx.EXPAND)
            self.wink_mode_choice.Bind(wx.EVT_CHOICE, self.change_wink_mode)

            separator = wx.StaticLine(self.panel, -1, size=(234, 3))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

        if True:
            iris_size_text = wx.StaticText(self.panel, label=" --- Iris Size --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(iris_size_text, 0, wx.EXPAND)

            self.iris_left_slider = wx.Slider(self.panel, minValue=0, maxValue=1000, value=0, style=wx.HORIZONTAL)
            self.panel_sizer.Add(self.iris_left_slider, 0, wx.EXPAND)
            self.iris_left_slider.Bind(wx.EVT_SLIDER, self.change_iris_size)

            self.iris_right_slider = wx.Slider(self.panel, minValue=0, maxValue=1000, value=0, style=wx.HORIZONTAL)
            self.panel_sizer.Add(self.iris_right_slider, 0, wx.EXPAND)
            self.iris_right_slider.Bind(wx.EVT_SLIDER, self.change_iris_size)
            self.iris_right_slider.Enable(False)

            self.link_left_right_irises = wx.CheckBox(
                self.panel, label="Use same value for both sides")
            self.link_left_right_irises.SetValue(True)
            self.panel_sizer.Add(self.link_left_right_irises, wx.SizerFlags().CenterHorizontal().Border())
            self.link_left_right_irises.Bind(wx.EVT_CHECKBOX, self.link_left_right_irises_clicked)

            separator = wx.StaticLine(self.panel, -1, size=(234, 3))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

        if True:
            breathing_frequency_text = wx.StaticText(
                self.panel, label=" --- Breathing --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(breathing_frequency_text, 0, wx.EXPAND)

            self.restart_breathing_cycle_button = wx.Button(self.panel, label="Restart Breathing Cycle")
            self.restart_breathing_cycle_button.Bind(wx.EVT_BUTTON, self.restart_breathing_cycle_clicked)
            self.panel_sizer.Add(self.restart_breathing_cycle_button, 0, wx.EXPAND)

            self.breathing_frequency_slider = wx.Slider(
                self.panel, minValue=0, maxValue=60, value=20, style=wx.HORIZONTAL)
            self.panel_sizer.Add(self.breathing_frequency_slider, 0, wx.EXPAND)

#            self.breathing_gauge = wx.Gauge(self.panel, style=wx.GA_HORIZONTAL, range=1000)
            self.breathing_gauge = wx.Gauge(self.panel, style=wx.GA_HORIZONTAL, range=1000, size=(234, 8))
            self.panel_sizer.Add(self.breathing_gauge, 0, wx.EXPAND)

#Pose calibration panel
            separator = wx.StaticLine(self.panel, -1, size=(234, 3))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

        if True:
            self.calibrate_front_position_button = wx.Button(self.panel, label="Calibrate Front Position")
            self.calibrate_front_position_button.Bind(wx.EVT_BUTTON, self.calibrate_front_position_clicked)
            self.panel_sizer.Add(self.calibrate_front_position_button, 0, wx.EXPAND)

            self.reset_front_position_button = wx.Button(self.panel, label="Reset Front Position")
            self.reset_front_position_button.Bind(wx.EVT_BUTTON, self.reset_front_position_clicked)
            self.panel_sizer.Add(self.reset_front_position_button, 0, wx.EXPAND)

            self.calibrate_head_x_text = wx.StaticText(
                self.panel, label=" --- Head_x --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(self.calibrate_head_x_text, 0, wx.EXPAND)

            self.calibrate_head_x_slider = wx.Slider(
                self.panel, minValue=-80, maxValue=80, value=0, style=wx.HORIZONTAL | wx.SL_LABELS)
            self.panel_sizer.Add(self.calibrate_head_x_slider, 0, wx.EXPAND)
            self.calibrate_head_x_slider.Bind(wx.EVT_SLIDER, self.change_head_x_slider)

            self.calibrate_head_y_text = wx.StaticText(
                self.panel, label=" --- Head_y --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(self.calibrate_head_y_text, 0, wx.EXPAND)

            self.calibrate_head_y_slider = wx.Slider(
                self.panel, minValue=-80, maxValue=80, value=0, style=wx.HORIZONTAL | wx.SL_LABELS)
            self.panel_sizer.Add(self.calibrate_head_y_slider, 0, wx.EXPAND)
            self.calibrate_head_y_slider.Bind(wx.EVT_SLIDER, self.change_head_y_slider)

            self.calibrate_head_z_text = wx.StaticText(
                self.panel, label=" --- Neck_z --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(self.calibrate_head_z_text, 0, wx.EXPAND)

            self.calibrate_head_z_slider = wx.Slider(
                self.panel, minValue=-80, maxValue=80, value=0, style=wx.HORIZONTAL | wx.SL_LABELS)
            self.panel_sizer.Add(self.calibrate_head_z_slider, 0, wx.EXPAND)
            self.calibrate_head_z_slider.Bind(wx.EVT_SLIDER, self.change_head_z_slider)

            separator = wx.StaticLine(self.panel, -1, size=(234, 3))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

#            self.weight_body_text = wx.StaticText(
#                self.panel, label=" Moving Rate of Body ", style=wx.ALIGN_CENTER)
#            self.panel_sizer.Add(self.weight_body_text, 0, wx.EXPAND)

            self.calibrate_body_y_text = wx.StaticText(
#                self.panel, label=" --- Body_y --- ", style=wx.ALIGN_CENTER)
                self.panel, label=" ---Moving Rate of Body_y --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(self.calibrate_body_y_text, 0, wx.EXPAND)

            self.calibrate_body_y_slider = wx.Slider(
                self.panel, minValue=-10, maxValue=10, value=0, style=wx.HORIZONTAL | wx.SL_LABELS)
            self.panel_sizer.Add(self.calibrate_body_y_slider, 0, wx.EXPAND)
            self.calibrate_body_y_slider.Bind(wx.EVT_SLIDER, self.change_body_y_slider)

            self.calibrate_body_z_text = wx.StaticText(
#                self.panel, label=" --- Body_z --- ", style=wx.ALIGN_CENTER)
                self.panel, label=" ---Moving Rate of Body_z --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(self.calibrate_body_z_text, 0, wx.EXPAND)

            self.calibrate_body_z_slider = wx.Slider(
                self.panel, minValue=-10, maxValue=10, value=0, style=wx.HORIZONTAL | wx.SL_LABELS)
            self.panel_sizer.Add(self.calibrate_body_z_slider, 0, wx.EXPAND)
            self.calibrate_body_z_slider.Bind(wx.EVT_SLIDER, self.change_body_z_slider)

            self.reset_body_button = wx.Button(self.panel, label="Fix Body")
            self.reset_body_button.Bind(wx.EVT_BUTTON, self.reset_body_clicked)
            self.panel_sizer.Add(self.reset_body_button, 0, wx.EXPAND)

        self.panel_sizer.Fit(self.panel)

    def restart_breathing_cycle_clicked(self, event: wx.Event):
        self.breathing_start_time = time.time()

    def change_sp_app(self, event: wx.Event):
        selected_index = self.sp_app_choice.GetSelection()
        if selected_index == 0:
            ifadd.SP_APP_MODE = SmartPhoneApp.IPHONE
        elif selected_index == 1:
            ifadd.SP_APP_MODE = SmartPhoneApp.ANDROID
        elif selected_index == 2:
            ifadd.SP_APP_MODE = SmartPhoneApp.VMC
        elif selected_index == 3:
            ifadd.SP_APP_MODE = SmartPhoneApp.VMCPERFECTSYNC
        elif selected_index == 4:
            ifadd.SP_APP_MODE = SmartPhoneApp.IFACIALMOCAPPC
        elif selected_index == 5:
            ifadd.SP_APP_MODE = SmartPhoneApp.VTUBESTUDIO
        else:
            ifadd.SP_APP_MODE = SmartPhoneApp.VMC

    def change_eyebrow_down_mode(self, event: wx.Event):
        selected_index = self.eyebrow_down_mode_choice.GetSelection()
        if selected_index == 0:
            self.args.eyebrow_down_mode = EyebrowDownMode.ANGRY
        elif selected_index == 1:
            self.args.eyebrow_down_mode = EyebrowDownMode.TROUBLED
        elif selected_index == 2:
            self.args.eyebrow_down_mode = EyebrowDownMode.SERIOUS
        else:
            self.args.eyebrow_down_mode = EyebrowDownMode.LOWERED

    def change_wink_mode(self, event: wx.Event):
        selected_index = self.wink_mode_choice.GetSelection()
        if selected_index == 0:
            self.args.wink_mode = WinkMode.NORMAL
        else:
            self.args.wink_mode = WinkMode.RELAXED

    def change_iris_size(self, event: wx.Event):
        if self.link_left_right_irises.GetValue():
            left_value = self.iris_left_slider.GetValue()
            right_value = self.iris_right_slider.GetValue()
            if left_value != right_value:
                self.iris_right_slider.SetValue(left_value)
            self.args.iris_small_left = left_value / 1000.0
            self.args.iris_small_right = left_value / 1000.0
        else:
            self.args.iris_small_left = self.iris_left_slider.GetValue() / 1000.0
            self.args.iris_small_right = self.iris_right_slider.GetValue() / 1000.0

    def link_left_right_irises_clicked(self, event: wx.Event):
        if self.link_left_right_irises.GetValue():
            self.iris_right_slider.Enable(False)
        else:
            self.iris_right_slider.Enable(True)
        self.change_iris_size(event)

    def calibrate_front_position_clicked(self, event: wx.Event):
        ifadd.CAL_HEAD_X = round(ifadd.POS_HEAD_X, 0)
        ifadd.CAL_HEAD_Y = round(ifadd.POS_HEAD_Y, 0)
        ifadd.CAL_HEAD_Z = round(ifadd.POS_HEAD_Z, 0)
        # ifadd.CAL_BODY_Y = ifadd.POS_BODY_Y
        # ifadd.CAL_BODY_Z = ifadd.POS_BODY_Z
        self.calibrate_head_x_slider.SetValue(int(ifadd.CAL_HEAD_X))
        self.calibrate_head_y_slider.SetValue(int(ifadd.CAL_HEAD_Y))
        self.calibrate_head_z_slider.SetValue(int(ifadd.CAL_HEAD_Z))
        # self.calibrate_body_y_slider.SetValue(ifadd.CAL_BODY_Y)
        # self.calibrate_body_z_slider.SetValue(ifadd.CAL_BODY_Z)

    def reset_front_position_clicked(self, event: wx.Event):
        ifadd.CAL_HEAD_X = 0.0
        ifadd.CAL_HEAD_Y = 0.0
        ifadd.CAL_HEAD_Z = 0.0
        # ifadd.CAL_BODY_Y = 0.0
        # ifadd.CAL_BODY_Z = 0.0
        self.calibrate_head_x_slider.SetValue(int(ifadd.CAL_HEAD_X))
        self.calibrate_head_y_slider.SetValue(int(ifadd.CAL_HEAD_Y))
        self.calibrate_head_z_slider.SetValue(int(ifadd.CAL_HEAD_Z))
        # self.calibrate_body_y_slider.SetValue(ifadd.CAL_BODY_Y)
        # self.calibrate_body_z_slider.SetValue(ifadd.CAL_BODY_Z)


    def change_head_x_slider(self, event: wx.Event):
        ifadd.CAL_HEAD_X = float(self.calibrate_head_x_slider.GetValue())
#        print(ifadd.CAL_HEAD_X)

    def change_head_y_slider(self, event: wx.Event):
        ifadd.CAL_HEAD_Y = float(self.calibrate_head_y_slider.GetValue())
#        print(ifadd.CAL_HEAD_Y)

    def change_head_z_slider(self, event: wx.Event):
        ifadd.CAL_HEAD_Z = float(self.calibrate_head_z_slider.GetValue())
#        print(ifadd.CAL_HEAD_Z)

    def change_body_y_slider(self, event: wx.Event):
        ifadd.CAL_BODY_Y = float(self.calibrate_body_y_slider.GetValue())
#        print(ifadd.CAL_BODY_Y)

    def change_body_z_slider(self, event: wx.Event):
        ifadd.CAL_BODY_Z = float(self.calibrate_body_z_slider.GetValue())
#        print(ifadd.CAL_BODY_Z)

    def reset_body_clicked(self, event: wx.Event):
        ifadd.CAL_BODY_Y = 0.0
        ifadd.CAL_BODY_Z = 0.0
        self.calibrate_body_y_slider.SetValue(int(ifadd.CAL_BODY_Y))
        self.calibrate_body_z_slider.SetValue(int(ifadd.CAL_BODY_Z))


    def decompose_head_body_param(self, param, threshold=2.0 / 3):
        if abs(param) < threshold:
            return (param, 0.0)
        else:
            if param < 0:
                sign = -1.0
            else:
                sign = 1.0
            return (threshold * sign, (abs(param) - threshold) * sign)

    def convert_old(self, ifacialmocap_pose: Dict[str, float]) -> List[float]:
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

    def convert(self, ifacialmocap_pose: Dict[str, float], isPerfectsync: bool) -> List[float]:
        if isPerfectsync == True:
            pose = self.convert_perfectsink(ifacialmocap_pose)
        else:
            pose = self.convert_vmc_vrm0(ifacialmocap_pose)
        
        return pose

    def convert_perfectsink(self, ifacialmocap_pose: Dict[str, float]) -> List[float]:
        pose = [0.0 for i in range(self.pose_size)]

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
            brow_inner_up = ifacialmocap_pose[BROW_INNER_UP]
            brow_outer_up_right = ifacialmocap_pose[BROW_OUTER_UP_RIGHT]
            brow_outer_up_left = ifacialmocap_pose[BROW_OUTER_UP_LEFT]

            brow_up_left = clamp(brow_inner_up + brow_outer_up_left, 0.0, 1.0)
            brow_up_right = clamp(brow_inner_up + brow_outer_up_right, 0.0, 1.0)
            pose[self.eyebrow_raised_left_index] = brow_up_left
            pose[self.eyebrow_raised_right_index] = brow_up_right

            brow_down_left = (1.0 - smile_degree) \
                             * clamp(ifacialmocap_pose[BROW_DOWN_LEFT] / self.args.eyebrow_down_max_value, 0.0, 1.0)
            brow_down_right = (1.0 - smile_degree) \
                              * clamp(ifacialmocap_pose[BROW_DOWN_RIGHT] / self.args.eyebrow_down_max_value, 0.0, 1.0)
            if self.args.eyebrow_down_mode == EyebrowDownMode.TROUBLED:
                pose[self.eyebrow_troubled_left_index] = brow_down_left
                pose[self.eyebrow_troubled_right_index] = brow_down_right
            elif self.args.eyebrow_down_mode == EyebrowDownMode.ANGRY:
                pose[self.eyebrow_angry_left_index] = brow_down_left
                pose[self.eyebrow_angry_right_index] = brow_down_right
            elif self.args.eyebrow_down_mode == EyebrowDownMode.LOWERED:
                pose[self.eyebrow_lowered_left_index] = brow_down_left
                pose[self.eyebrow_lowered_right_index] = brow_down_right
            elif self.args.eyebrow_down_mode == EyebrowDownMode.SERIOUS:
                pose[self.eyebrow_serious_left_index] = brow_down_left
                pose[self.eyebrow_serious_right_index] = brow_down_right

            brow_happy_value = clamp(smile_value, 0.0, 1.0) * smile_degree
            pose[self.eyebrow_happy_left_index] = brow_happy_value
            pose[self.eyebrow_happy_right_index] = brow_happy_value

        # Eye
        if True:
            # Surprised
            pose[self.eye_surprised_left_index] = clamp(
                ifacialmocap_pose[EYE_WIDE_LEFT] / self.args.eye_wide_max_value, 0.0, 1.0)
            pose[self.eye_surprised_right_index] = clamp(
                ifacialmocap_pose[EYE_WIDE_RIGHT] / self.args.eye_wide_max_value, 0.0, 1.0)

            # Wink
            if self.args.wink_mode == WinkMode.NORMAL:
                wink_left_index = self.eye_wink_left_index
                wink_right_index = self.eye_wink_right_index
            else:
                wink_left_index = self.eye_relaxed_left_index
                wink_right_index = self.eye_relaxed_right_index
            pose[wink_left_index] = (1.0 - smile_degree) * clamp(
                ifacialmocap_pose[EYE_BLINK_LEFT] / self.args.eye_blink_max_value, 0.0, 1.0)
            pose[wink_right_index] = (1.0 - smile_degree) * clamp(
                ifacialmocap_pose[EYE_BLINK_RIGHT] / self.args.eye_blink_max_value, 0.0, 1.0)
            pose[self.eye_happy_wink_left_index] = smile_degree * clamp(
                ifacialmocap_pose[EYE_BLINK_LEFT] / self.args.eye_blink_max_value, 0.0, 1.0)
            pose[self.eye_happy_wink_right_index] = smile_degree * clamp(
                ifacialmocap_pose[EYE_BLINK_RIGHT] / self.args.eye_blink_max_value, 0.0, 1.0)

            # Lower eyelid
            cheek_squint_denom = self.args.cheek_squint_max_value - self.args.cheek_squint_min_value
            pose[self.eye_raised_lower_eyelid_left_index] = \
                clamp(
                    (ifacialmocap_pose[CHEEK_SQUINT_LEFT] - self.args.cheek_squint_min_value) / cheek_squint_denom,
                    0.0, 1.0)
            pose[self.eye_raised_lower_eyelid_right_index] = \
                clamp(
                    (ifacialmocap_pose[CHEEK_SQUINT_RIGHT] - self.args.cheek_squint_min_value) / cheek_squint_denom,
                    0.0, 1.0)

        # Iris rotation
        if True:
            eye_rotation_y = (ifacialmocap_pose[EYE_LOOK_IN_LEFT]
                              - ifacialmocap_pose[EYE_LOOK_OUT_LEFT]
                              - ifacialmocap_pose[EYE_LOOK_IN_RIGHT]
                              + ifacialmocap_pose[EYE_LOOK_OUT_RIGHT]) / 2.0 * self.args.eye_rotation_factor
            pose[self.iris_rotation_y_index] = clamp(eye_rotation_y, -1.0, 1.0)

            eye_rotation_x = (ifacialmocap_pose[EYE_LOOK_UP_LEFT]
                              + ifacialmocap_pose[EYE_LOOK_UP_RIGHT]
                              - ifacialmocap_pose[EYE_LOOK_DOWN_LEFT]
                              - ifacialmocap_pose[EYE_LOOK_DOWN_RIGHT]) / 2.0 * self.args.eye_rotation_factor
            pose[self.iris_rotation_x_index] = clamp(eye_rotation_x, -1.0, 1.0)

        # Iris size
        if True:
            pose[self.iris_small_left_index] = self.args.iris_small_left
            pose[self.iris_small_right_index] = self.args.iris_small_right

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
            pose[self.head_x_index] = x_param

            y_param = clamp(-(ifadd.POS_HEAD_Y-ifadd.CAL_HEAD_Y), -10.0, 10.0) / 10.0
            pose[self.head_y_index] = y_param
            pose[self.body_y_index] = y_param * ifadd.CAL_BODY_Y / 10.0

            z_param = clamp((ifadd.POS_HEAD_Z-ifadd.CAL_HEAD_Z), -15.0, 15.0) / 15.0
            pose[self.neck_z_index] = z_param
            pose[self.body_z_index] = z_param * ifadd.CAL_BODY_Z / 10.0

            # Mouth
        if True:
            jaw_open_denom = self.args.jaw_open_max_value - self.args.jaw_open_min_value
            mouth_open = clamp((ifacialmocap_pose[JAW_OPEN] - self.args.jaw_open_min_value) / jaw_open_denom, 0.0, 1.0)
            pose[self.mouth_aaa_index] = mouth_open
            pose[self.mouth_raised_corner_left_index] = clamp(smile_value, 0.0, 1.0)
            pose[self.mouth_raised_corner_right_index] = clamp(smile_value, 0.0, 1.0)

            is_mouth_open = mouth_open > 0.0
            if not is_mouth_open:
                mouth_frown_value = clamp(
                    (ifacialmocap_pose[MOUTH_FROWN_LEFT] + ifacialmocap_pose[
                        MOUTH_FROWN_RIGHT]) / self.args.mouth_frown_max_value, 0.0, 1.0)
                pose[self.mouth_lowered_corner_left_index] = mouth_frown_value
                pose[self.mouth_lowered_corner_right_index] = mouth_frown_value
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
                pose[self.mouth_aaa_index] = restricted_decomp[0]
                pose[self.mouth_iii_index] = restricted_decomp[1]
                mouth_funnel_denom = self.args.mouth_funnel_max_value - self.args.mouth_funnel_min_value
                ooo_alpha = clamp((mouth_funnel - self.args.mouth_funnel_min_value) / mouth_funnel_denom, 0.0, 1.0)
                uo_value = clamp(restricted_decomp[2] + restricted_decomp[3], 0.0, 1.0)
                pose[self.mouth_uuu_index] = uo_value * (1.0 - ooo_alpha)
                pose[self.mouth_ooo_index] = uo_value * ooo_alpha

        if self.panel is not None:
            frequency = self.breathing_frequency_slider.GetValue()
            if frequency == 0:
                value = 0.0
                pose[self.breathing_index] = value
                self.breathing_start_time = time.time()
            else:
                period = 60.0 / frequency
                now = time.time()
                diff = now - self.breathing_start_time
                frac = (diff % period) / period
                value = (-math.cos(2 * math.pi * frac) + 1.0) / 2.0
                pose[self.breathing_index] = value
            self.breathing_gauge.SetValue(int(1000 * value))

        return pose

    def convert_vmc_vrm0(self, ifacialmocap_pose: Dict[str, float]) -> List[float]:
        pose = [0.0 for i in range(self.pose_size)]

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
            brow_inner_up = 0.0 # ifacialmocap_pose[BROW_INNER_UP]
            brow_outer_up_right = 0.0 # ifacialmocap_pose[BROW_OUTER_UP_RIGHT]
            brow_outer_up_left = 0.0 # ifacialmocap_pose[BROW_OUTER_UP_LEFT]

            brow_up_left = clamp(brow_inner_up + brow_outer_up_left, 0.0, 1.0)
            brow_up_right = clamp(brow_inner_up + brow_outer_up_right, 0.0, 1.0)
            pose[self.eyebrow_raised_left_index] = brow_up_left
            pose[self.eyebrow_raised_right_index] = brow_up_right

            # brow_down_left = (1.0 - smile_degree) \
            #                  * clamp(ifacialmocap_pose[BROW_DOWN_LEFT] / self.args.eyebrow_down_max_value, 0.0, 1.0)
            # brow_down_right = (1.0 - smile_degree) \
            #                   * clamp(ifacialmocap_pose[BROW_DOWN_RIGHT] / self.args.eyebrow_down_max_value, 0.0, 1.0)
            brow_down_left = 0.0
            brow_down_right = 0.0

            if self.args.eyebrow_down_mode == EyebrowDownMode.TROUBLED:
                pose[self.eyebrow_troubled_left_index] = brow_down_left
                pose[self.eyebrow_troubled_right_index] = brow_down_right
            elif self.args.eyebrow_down_mode == EyebrowDownMode.ANGRY:
                pose[self.eyebrow_angry_left_index] = brow_down_left
                pose[self.eyebrow_angry_right_index] = brow_down_right
            elif self.args.eyebrow_down_mode == EyebrowDownMode.LOWERED:
                pose[self.eyebrow_lowered_left_index] = brow_down_left
                pose[self.eyebrow_lowered_right_index] = brow_down_right
            elif self.args.eyebrow_down_mode == EyebrowDownMode.SERIOUS:
                pose[self.eyebrow_serious_left_index] = brow_down_left
                pose[self.eyebrow_serious_right_index] = brow_down_right

            brow_happy_value = clamp(smile_value, 0.0, 1.0) * smile_degree
            pose[self.eyebrow_happy_left_index] = brow_happy_value
            pose[self.eyebrow_happy_right_index] = brow_happy_value

        # Eye
        if True:
            # Surprised
            pose[self.eye_surprised_left_index] = clamp(
                ifacialmocap_pose["Surprised"] / self.args.eye_wide_max_value, 0.0, 1.0)
            pose[self.eye_surprised_right_index] = clamp(
                ifacialmocap_pose["Surprised"] / self.args.eye_wide_max_value, 0.0, 1.0)

            # Wink
            if self.args.wink_mode == WinkMode.NORMAL:
                wink_left_index = self.eye_wink_left_index
                wink_right_index = self.eye_wink_right_index
            else:
                wink_left_index = self.eye_relaxed_left_index
                wink_right_index = self.eye_relaxed_right_index
            pose[wink_left_index] = (1.0 - smile_degree) * clamp(
                (ifacialmocap_pose["Blink_L"] + ifacialmocap_pose["Blink"]) / self.args.eye_blink_max_value, 0.0, 1.0)
            pose[wink_right_index] = (1.0 - smile_degree) * clamp(
                (ifacialmocap_pose["Blink_R"] + ifacialmocap_pose["Blink"]) / self.args.eye_blink_max_value, 0.0, 1.0)
            pose[self.eye_happy_wink_left_index] = smile_degree * clamp(
                (ifacialmocap_pose["Blink_L"] + ifacialmocap_pose["Blink"]) / self.args.eye_blink_max_value, 0.0, 1.0)
            pose[self.eye_happy_wink_right_index] = smile_degree * clamp(
                (ifacialmocap_pose["Blink_R"] + ifacialmocap_pose["Blink"]) / self.args.eye_blink_max_value, 0.0, 1.0)

            # Lower eyelid
            cheek_squint_denom = self.args.cheek_squint_max_value - self.args.cheek_squint_min_value
            # pose[self.eye_raised_lower_eyelid_left_index] = \
            #     clamp(
            #         (ifacialmocap_pose[CHEEK_SQUINT_LEFT] - self.args.cheek_squint_min_value) / cheek_squint_denom,
            #         0.0, 1.0)
            # pose[self.eye_raised_lower_eyelid_right_index] = \
            #     clamp(
            #         (ifacialmocap_pose[CHEEK_SQUINT_RIGHT] - self.args.cheek_squint_min_value) / cheek_squint_denom,
            #         0.0, 1.0)
            pose[self.eye_raised_lower_eyelid_left_index] = 0.0
            pose[self.eye_raised_lower_eyelid_right_index] = 0.0

        # Iris rotation
        if True:
            eye_rotation_y = (ifacialmocap_pose[RIGHT_EYE_BONE_Y]
                              + ifacialmocap_pose[LEFT_EYE_BONE_Y]) / 2.0 * self.args.eye_rotation_factor
            pose[self.iris_rotation_y_index] = clamp(eye_rotation_y, -1.0, 1.0)

            eye_rotation_x = (ifacialmocap_pose[RIGHT_EYE_BONE_X]
                              + ifacialmocap_pose[LEFT_EYE_BONE_X]) / 2.0 * self.args.eye_rotation_factor
            pose[self.iris_rotation_x_index] = clamp(eye_rotation_x, -1.0, 1.0)

        # Iris size
        if True:
            pose[self.iris_small_left_index] = self.args.iris_small_left
            pose[self.iris_small_right_index] = self.args.iris_small_right

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
            pose[self.head_x_index] = x_param

            y_param = clamp(-(ifadd.POS_HEAD_Y-ifadd.CAL_HEAD_Y), -10.0, 10.0) / 10.0
            pose[self.head_y_index] = y_param
            pose[self.body_y_index] = y_param * ifadd.CAL_BODY_Y / 10.0

            z_param = clamp((ifadd.POS_HEAD_Z-ifadd.CAL_HEAD_Z), -15.0, 15.0) / 15.0
            pose[self.neck_z_index] = z_param
            pose[self.body_z_index] = z_param * ifadd.CAL_BODY_Z / 10.0

            # Mouth
        if True:
            # jaw_open_denom = self.args.jaw_open_max_value - self.args.jaw_open_min_value
            # mouth_open = clamp((ifacialmocap_pose[JAW_OPEN] - self.args.jaw_open_min_value) / jaw_open_denom, 0.0, 1.0)
            # pose[self.mouth_aaa_index] = ifacialmocap_pose["A"]
            pose[self.mouth_raised_corner_left_index] = clamp(smile_value, 0.0, 1.0)
            pose[self.mouth_raised_corner_right_index] = clamp(smile_value, 0.0, 1.0)

            pose[self.mouth_lowered_corner_left_index] = 0.0
            pose[self.mouth_lowered_corner_right_index] = 0.0

            pose[self.mouth_aaa_index] = clamp(ifacialmocap_pose["A"], 0.0, 1.0)
            pose[self.mouth_iii_index] = clamp(ifacialmocap_pose["I"], 0.0, 1.0)
            pose[self.mouth_uuu_index] = clamp(ifacialmocap_pose["U"], 0.0, 1.0)
            pose[self.mouth_eee_index] = clamp(ifacialmocap_pose["E"], 0.0, 1.0)
            pose[self.mouth_ooo_index] = clamp(ifacialmocap_pose["O"], 0.0, 1.0)

        if self.panel is not None:
            frequency = self.breathing_frequency_slider.GetValue()
            if frequency == 0:
                value = 0.0
                pose[self.breathing_index] = value
                self.breathing_start_time = time.time()
            else:
                period = 60.0 / frequency
                now = time.time()
                diff = now - self.breathing_start_time
                frac = (diff % period) / period
                value = (-math.cos(2 * math.pi * frac) + 1.0) / 2.0
                pose[self.breathing_index] = value
            self.breathing_gauge.SetValue(int(1000 * value))

        return pose

def create_ifacialmocap_pose_converter(
        args: Optional[IFacialMocapPoseConverter25Args] = None) -> IFacialMocapPoseConverter:
    return IFacialMocapPoseConverter25(args)
