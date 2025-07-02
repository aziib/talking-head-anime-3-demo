import argparse
import os
import socket
import sys
import threading
import time
import imageio.v2 as imageio
import numpy as np
import http.server
import socketserver
from typing import Optional
from PIL import Image
from flask import Flask, send_file
from flask import Flask, Response
from flask_cors import CORS
import io
import asyncio
import websockets

sys.path.append(os.getcwd())

from tha3.mocap.ifacialmocap_pose import create_default_ifacialmocap_pose
# Unused iFacialMocap V2 specific imports commented out or removed
# from tha3.mocap.ifacialmocap_v2 import IFACIALMOCAP_PORT, IFACIALMOCAP_START_STRING, parse_ifacialmocap_v2_pose, \
#     parse_ifacialmocap_v1_pose, parse_meowface_pose, parse_vmc_pose, parse_vts_pose, parse_vmc_pose_list, parse_vmc_perfectsync_pose_list
from tha3.poser.modes.load_poser import load_poser
from tha3.mocap.mediapipe_input import MediaPipeWebcamInput # Added

from tha3.mocap.ifacialmocap_poser_converter_25 import SmartPhoneApp # This might be less relevant or need adaptation

import torch
import wx
import wx.adv
import json
import PIL.Image

# import pdb

from tha3.poser.poser import Poser
from tha3.mocap.ifacialmocap_constants import *
from tha3.mocap.ifacialmocap_pose_converter import IFacialMocapPoseConverter
from tha3.util import torch_linear_to_srgb, resize_PIL_image, extract_PIL_image_from_filelike, \
    extract_pytorch_image_from_PIL_image


import tha3.mocap.ifacialmocap_add as ifadd


def convert_linear_to_srgb(image: torch.Tensor) -> torch.Tensor:
    image = image.to(device)
    rgb_image = torch_linear_to_srgb(image[0:3, :, :])
    return torch.cat([rgb_image, image[3:4, :, :]], dim=0)


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


class MainFrame(wx.Frame):
    def __init__(self, poser: Poser, pose_converter: IFacialMocapPoseConverter, device:torch.device, mocap_port = 49983):
#        super().__init__(None, wx.ID_ANY, "iFacialMocap Puppeteer (Marigold)")
        super().__init__(None, wx.ID_ANY, "MediaPipe Face Puppeteer (Talking Head Anime 3)") # Changed Window Title
        self.pose_converter = pose_converter
        self.poser = poser
        self.device = device

        self.mediapipe_input = None # Added
        self.is_mediapipe_running = False # Added

        self.is_recording = False
        self.temp_folder = "temp"
        self.output_folder = "output"
        self.stream_folder = "stream"
        self.record_timer = None
        self.stream_timer = None
        self.record_counter = 0
        self.stream_counter = 0
        self.flask_thread = None
        self.image_save_counter = 0
        self.last_output_numpy_image = None


        self.mocap_port = mocap_port

        self.ifacialmocap_pose = create_default_ifacialmocap_pose()
        self.source_image_bitmap = wx.Bitmap(self.poser.get_image_size(), self.poser.get_image_size())
        self.result_image_bitmap = wx.Bitmap(self.poser.get_image_size(), self.poser.get_image_size())
        self.wx_source_image = None
        self.torch_source_image = None
        self.last_pose = None
        self.fps_statistics = FpsStatistics()
        self.last_update_time = None
        self.same_pose_count = 0

        # self.last_torch_image = torch.zeros(4, 512, 512).float()
        self.last_torch_image = None
        self.source_image_string = None

        self.last_show_index = -1
        self.last_output_index = -1

        self.vts_already_request = False
        self.vts_ip = "192.168.0.1"
        self.vts_port = 21412 # Likely not needed for MediaPipe

        self.vmc_cache = [] # Likely not needed for MediaPipe
        self.poseIsPerfectsync = False # This might still be relevant depending on how pose_converter is used

        # self.create_receiving_socket() # Removed for MediaPipe
        self.create_ui() # UI creation will be adapted later
        self.create_timers() # Timer creation will be adapted later
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.read_config_file() # Config reading will be adapted later

        self.update_source_image_bitmap()
        self.update_result_image_bitmap()

    # def create_receiving_socket(self): # Removed for MediaPipe
    #     self.receiving_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #     # self.receiving_socket.bind(("", IFACIALMOCAP_PORT))
    #     self.receiving_socket.bind(("", self.mocap_port))
    #     self.receiving_socket.setblocking(False)

    def create_timers(self):
        # self.capture_timer = wx.Timer(self, wx.ID_ANY) # This timer updated the raw iFacialMocap values display
        # self.Bind(wx.EVT_TIMER, self.update_capture_panel, id=self.capture_timer.GetId()) # We might not need this panel for MediaPipe
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
            capture_ip = config_json_dict.get("capture_ip", "192.168.0.1") # Less relevant for MediaPipe
            # mocap_method = config_json_dict.get("mocap_method", 2) # This relates to iFacialMocap app type, less relevant
            webcam_index_str = config_json_dict.get("webcam_index", "0") # Added for MediaPipe
            self.webcam_index_text_ctrl.SetValue(webcam_index_str) # Added for MediaPipe UI element

            eyebrow_mode = config_json_dict.get("eyebrow_mode", 0)
            wink_mode = config_json_dict.get("wink_mode", 0)
            irissize_left = config_json_dict.get("irissize_left", 0)
            irissize_right = config_json_dict.get("irissize_right", 0)
            irissize_link = config_json_dict.get("irissize_link", True)
            breathing = config_json_dict.get("breathing", 20)
            head_x = config_json_dict.get("head_x", 0.0)
            head_y = config_json_dict.get("head_y", 0.0)
            neck_z = config_json_dict.get("neck_z", 0.0)
            body_y = config_json_dict.get("body_y", 0.0)
            body_z = config_json_dict.get("body_z", 0.0)
            backgroud = config_json_dict.get("backgroud", 0)

            image_list = config_json_dict.get("image_list_512", [])
            image_select = config_json_dict.get("image_select_512", -1)
            image_output = config_json_dict.get("image_output_512", -1)

            self.capture_device_ip_text_ctrl.SetValue(capture_ip) # Keep for now, might remove/hide later

            # self.pose_converter.sp_app_choice.SetSelection(mocap_method) # Less relevant for MediaPipe
            # self.pose_converter.change_sp_app(self) # Less relevant
            self.pose_converter.eyebrow_down_mode_choice.SetSelection(eyebrow_mode)
            self.pose_converter.change_eyebrow_down_mode(self)
            self.pose_converter.wink_mode_choice.SetSelection(wink_mode)
            self.pose_converter.change_wink_mode(self)

            self.pose_converter.iris_left_slider.SetValue(irissize_left)
            self.pose_converter.iris_right_slider.SetValue(irissize_right)
            self.pose_converter.link_left_right_irises.SetValue(irissize_link)
            self.pose_converter.link_left_right_irises_clicked(self)
            self.pose_converter.change_iris_size(self)

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
                    try:
                        image_name = os.path.basename(image_file_name)
                        pil_image = resize_PIL_image(
                            extract_PIL_image_from_filelike(image_file_name),
                            (self.poser.get_image_size(), self.poser.get_image_size()))
                        w, h = pil_image.size
                        if pil_image.mode != 'RGBA':
                            raise Exception("Image must have alpha channel!")
                        else:
                            wx_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                            torch_image = extract_pytorch_image_from_PIL_image(pil_image)
                    except Exception as e:
                        print(e)
                        image_name = image_name + "  (Image Loading Error!)"
                        w, h = 512, 512
                        pil_image = PIL.Image.new("RGBA", (w, h), (0, 0, 0, 0))
                        wx_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                        torch_image = extract_pytorch_image_from_PIL_image(pil_image)
                    self.source_image_list.Append(image_name, [pil_image, wx_image, torch_image, image_file_name])

                if image_list_index <= image_select:
                    image_select = -1
                if image_list_index <= image_output:
                    image_output = -1
                if image_output >= 0:
                    if image_select < 0:
                        image_select = image_output
                    self.last_output_index = image_output
                    image_sets = self.source_image_list.GetClientData(image_output)
                    self.torch_source_image = image_sets[2].to(self.device).to(self.poser.get_dtype())
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
                # capture_ip = self.capture_device_ip_text_ctrl.GetValue() # Commented out: No longer primary config for MediaPipe
                # mocap_method = self.pose_converter.sp_app_choice.GetSelection() # Commented out: No longer primary config for MediaPipe
                webcam_idx_str = self.webcam_index_text_ctrl.GetValue() # Added
                config_json_dict["webcam_index"] = webcam_idx_str # Added for MediaPipe

                eyebrow_mode = self.pose_converter.eyebrow_down_mode_choice.GetSelection()
                wink_mode = self.pose_converter.wink_mode_choice.GetSelection()
                irissize_left = self.pose_converter.iris_left_slider.GetValue()
                irissize_right = self.pose_converter.iris_right_slider.GetValue()
                irissize_link = self.pose_converter.link_left_right_irises.GetValue()
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
                    image_fullpath = image_sets[3]
                    image_list.append(image_fullpath)
                image_select = self.last_show_index
                image_output = self.last_output_index

                # config_json_dict["capture_ip"] = capture_ip # Commented out
                # config_json_dict["mocap_method"] = mocap_method # Commented out
                config_json_dict["eyebrow_mode"] = eyebrow_mode
                config_json_dict["wink_mode"] = wink_mode
                config_json_dict["irissize_left"] = irissize_left
                config_json_dict["irissize_right"] = irissize_right
                config_json_dict["irissize_link"] = irissize_link
                config_json_dict["breathing"] = breathing
                config_json_dict["head_x"] = head_x
                config_json_dict["head_y"] = head_y
                config_json_dict["neck_z"] = neck_z
                config_json_dict["body_y"] = body_y
                config_json_dict["body_z"] = body_z
                config_json_dict["backgroud"] = backgroud

                config_json_dict["image_list_512"] = image_list
                config_json_dict["image_select_512"] = image_select
                config_json_dict["image_output_512"] = image_output

                with open('tha3sw_config.json', 'w') as f:
                    json.dump(config_json_dict, f, indent=4)
        except Exception as e:
            print (e)

    def on_close(self, event: wx.Event):
        # Stop the timers
        self.animation_timer.Stop()
        # if self.capture_timer: self.capture_timer.Stop() # If capture_timer is used

        if self.mediapipe_input: # Added
            self.mediapipe_input.stop_capture() # Added
            self.is_mediapipe_running = False

        # Save config file
        self.save_config_file()

        # Close receiving socket # Removed
        # self.receiving_socket.close()

        # Destroy the windows
        self.Destroy()
        event.Skip()

    def on_start_capture(self, event: wx.Event): # Re-purposed for MediaPipe
        if self.is_mediapipe_running:
            if self.mediapipe_input:
                self.mediapipe_input.stop_capture()
            self.is_mediapipe_running = False
            self.start_capture_button.SetLabel("START WEBCAM")
            self.show_status_indicator(False, False)
            if self.animation_timer.IsRunning(): # Stop animation if webcam stops
                self.animation_timer.Stop()
                self.fps_text.SetLabelText("FPS = 0.00")

        else:
            webcam_idx_str = self.webcam_index_text_ctrl.GetValue()
            try:
                webcam_idx = int(webcam_idx_str)
            except ValueError:
                wx.MessageBox("Invalid Webcam Index. Please enter a number (e.g., 0).", "Error", wx.OK | wx.ICON_ERROR)
                return

            if self.mediapipe_input is None:
                self.mediapipe_input = MediaPipeWebcamInput(webcam_index=webcam_idx)
            else: # Ensure webcam index is updated if changed
                self.mediapipe_input.webcam_index = webcam_idx


            if self.mediapipe_input.start_capture():
                self.is_mediapipe_running = True
                self.start_capture_button.SetLabel("STOP WEBCAM")
                self.show_status_indicator(True, False) # True for "trying to receive", status to be updated by frame processing
                if not self.animation_timer.IsRunning():
                    maintimer = 20 # Default, or read from config
                    if os.path.exists('tha3sw_config.json'):
                        try:
                            with open('tha3sw_config.json') as f: config_json_dict = json.load(f)
                            maintimer = int(config_json_dict.get("timer", 20))
                        except: pass
                    self.animation_timer.Start(maintimer)
            else:
                wx.MessageBox(f"Failed to start webcam index {webcam_idx}.", "Error", wx.OK | wx.ICON_ERROR)
                self.show_status_indicator(False, False)
                self.mediapipe_input = None # Reset if failed to start


    # def vts_send_request(self): # Removed, not relevant for MediaPipe
    #     pass


    def read_mediapipe_pose(self): # Renamed and changed from read_ifacialmocap_pose
        if not self.is_mediapipe_running or self.mediapipe_input is None:
            self.show_status_indicator(False, False)
            return create_default_ifacialmocap_pose()

        # The process_frame in mediapipe_input.py now returns a tuple: (image, data_dict_or_string)
        processed_image, pose_data = self.mediapipe_input.process_frame(
            show_video=self.show_mediapipe_debug_window_checkbox.IsChecked()
        )

        if pose_data == "STOP":
            # This logic means the user closed the debug window using ESC
            # We should simulate clicking the stop button for our wx UI
            wx.CallAfter(self.on_start_capture, None) # Use CallAfter to avoid issues from event handler
            return create_default_ifacialmocap_pose()

        if pose_data and isinstance(pose_data, dict):
            self.ifacialmocap_pose = pose_data
            face_detected = self.mediapipe_input.raw_landmarks is not None
            self.show_status_indicator(True, face_detected)
            # Perfect sync is generally true for direct landmark tracking if we map all required blendshapes
            self.poseIsPerfectsync = True # Assume perfect sync capability for now
        elif processed_image is None and pose_data is None: # Indicates an issue or end of stream from webcam
            self.show_status_indicator(True, False) # Still "running" but no data
            print("Warning: No image or data from MediaPipe process_frame.")
        else: # No face detected in a valid frame or other non-STOP non-dict return
            self.show_status_indicator(True, False)
            # self.ifacialmocap_pose remains the last valid or default pose

        return self.ifacialmocap_pose

    def on_erase_background(self, event: wx.Event):
        pass

    def create_animation_panel(self, parent):
        self.animation_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.animation_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.animation_panel.SetSizer(self.animation_panel_sizer)
        self.animation_panel.SetAutoLayout(1)

        image_size = self.poser.get_image_size()

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

        # Add Webcam Index UI elements
        webcam_index_text = wx.StaticText(self.connection_panel, label="Webcam Index:", style=wx.ALIGN_RIGHT)
        self.connection_panel_sizer.Add(webcam_index_text, wx.SizerFlags(0).FixedMinSize().CenterVertical().Border(wx.ALL, 3))
        self.webcam_index_text_ctrl = wx.TextCtrl(self.connection_panel, value="0", size=(40,-1))
        self.connection_panel_sizer.Add(self.webcam_index_text_ctrl, wx.SizerFlags(0).FixedMinSize().CenterVertical().Border(wx.ALL, 3))

        self.show_mediapipe_debug_window_checkbox = wx.CheckBox(self.connection_panel, label="Show MP Debug")
        self.connection_panel_sizer.Add(self.show_mediapipe_debug_window_checkbox, wx.SizerFlags(0).FixedMinSize().CenterVertical().Border(wx.ALL, 3))
        self.show_mediapipe_debug_window_checkbox.SetValue(False) # Default to not showing external window
        self.show_mediapipe_debug_window_checkbox.Bind(wx.EVT_CHECKBOX, self.on_toggle_mp_debug_window)

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

        # Flexible spacer to push subsequent items to the right
        self.connection_panel_sizer.AddStretchSpacer(prop=1)

        self.reset_ok_button.Disable()
        self.reset_cancel_button.Disable()

        self.snapshot_button = wx.Button(self.connection_panel, label="Snapshot")
        self.connection_panel_sizer.Add(self.snapshot_button, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.snapshot_button.Bind(wx.EVT_BUTTON, self.on_snapshot)

        self.record_button = wx.Button(self.connection_panel, label="Record")
        self.connection_panel_sizer.Add(self.record_button, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.record_button.Bind(wx.EVT_BUTTON, self.on_record)

        self.stop_button = wx.Button(self.connection_panel, label="Stop")
        self.connection_panel_sizer.Add(self.stop_button, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.stop_button.Bind(wx.EVT_BUTTON, self.on_stop)

        self.stream_button = wx.Button(self.connection_panel, label="Stream")
        self.connection_panel_sizer.Add(self.stream_button, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.stream_button.Bind(wx.EVT_BUTTON, self.stream_image)


        # Capture Device IP related UI - Not needed for MediaPipe, so we remove it or make it non-functional.
        # self.capture_device_ip_text_ctrl is still created for config compatibility but not added to sizer here.
        # It's set to "N/A (MediaPipe)" and read-only in its declaration if we decide to show it.
        # For now, it's not added to this panel's sizer.

        self.start_capture_button = wx.Button(self.connection_panel, label="START WEBCAM") # Changed label
        self.connection_panel_sizer.Add(self.start_capture_button, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.start_capture_button.Bind(wx.EVT_BUTTON, self.on_start_capture) # Method is now repurposed

        capture_status_text = wx.StaticText(self.connection_panel, label="MP Status:", style=wx.ALIGN_RIGHT) # Changed label
        self.connection_panel_sizer.Add(capture_status_text, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))

        self.capture_status_indicator = wx.StaticText(self.connection_panel, label=" ● ", style=wx.ALIGN_RIGHT)
        self.connection_panel_sizer.Add(self.capture_status_indicator, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.show_status_indicator(False, False)

    def on_toggle_mp_debug_window(self, event):
        # This event is just for noting the checkbox state change.
        # The actual use of the checkbox is in read_mediapipe_pose()
        # and when stopping mediapipe capture to close the window.
        if not self.show_mediapipe_debug_window_checkbox.IsChecked() and self.mediapipe_input:
            # If unchecked, tell mediapipe_input to close its window if it's managing one.
            # This requires a method in MediaPipeWebcamInput, e.g., close_debug_window()
            # For now, the window is managed by process_frame's cv2.waitKey.
            # If the debug window is managed by cv2.imshow in a loop within process_frame,
            # simply not calling imshow (because show_video is false) will hide it.
            pass

    # Update the on_snapshot function in ifacialmocap_puppeteer.py
    def on_snapshot(self, event: wx.Event):
        output_dir = "output"
        if not os.path.exists(output_dir):
           os.makedirs(output_dir)

        # Generate new file name
        new_file_number = len(os.listdir(output_dir))
        new_file_name = f"{new_file_number}.png"
        new_file_path = os.path.join(output_dir, new_file_name)

        # Save snapshot with transparency
        try:
            image_file_name = f"output/image_{self.image_save_counter:04d}.png"
            self.save_last_numpy_image(image_file_name)
            self.image_save_counter += 1  # Increment the counter after saving
            print(f"Image saved quickly as: {image_file_name}")
        except IOError:
            wx.LogError(f"Can't save file '{new_file_path}'.")

    def save_last_numpy_image(self, image_file_name):
        numpy_image = self.last_output_numpy_image  # Convert PyTorch Tensor to NumPy array
        pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA')
        os.makedirs(os.path.dirname(image_file_name), exist_ok=True)
        pil_image.save(image_file_name)

    def on_record(self, event):
        self.is_recording = True
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        else:
            # Delete all files in the temp folder
            for filename in os.listdir(self.temp_folder):
                file_path = os.path.join(self.temp_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

        self.record_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.take_snapshot, self.record_timer)
        frame_interval_ms = 1
        self.record_timer.Start(frame_interval_ms)  # Start the timer to take a snapshot every ms

    def take_snapshot(self, event):
        # Take snapshot and save in temp folder
        image_file_name = os.path.join(self.temp_folder, f"snapshot_{self.record_counter}.png")
        self.save_last_numpy_image(image_file_name)
        print(f"Image saved quickly as: {image_file_name}")
        # Code to take snapshot and save the image

        self.record_counter += 1

    def take_stream(self, event):
        # Take snapshot and save in temp folder
        image_file_name = os.path.join(self.stream_folder, f"stream.png")
        self.save_last_numpy_image(image_file_name)
        # Code to take snapshot and save the image

        self.record_counter += 1

    def on_stop(self, event):
        self.is_recording = False
        if self.record_timer:
            self.record_timer.Stop()
        print("Recording stopped.")
        self.record_counter = 0

        # Combine images into an APNG
        images = []
        temp_folder_path = os.path.join(os.getcwd(), self.temp_folder)
        output_file_path = os.path.join(os.getcwd(), self.output_folder, "recorded_animation.apng")

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Check if the file already exists and modify the name if it does
        base_name, extension = os.path.splitext(output_file_path)
        counter = 1
        while os.path.isfile(output_file_path):
            output_file_path = f"{base_name}_{counter}{extension}"
            counter += 1

        # Load images from the temporary folder
        for filename in sorted(os.listdir(temp_folder_path), key=lambda x: int(x.split('_')[1].split('.')[0])):
            file_path = os.path.join(temp_folder_path, filename)
            images.append(imageio.imread(file_path))

        # Save the images as an APNG
        imageio.mimsave(output_file_path, images, format='APNG', fps=12)  # Adjust fps to desired frame rate

        print(f"APNG saved to {output_file_path}")

        # Clean up the temporary folder
        for filename in os.listdir(temp_folder_path):
            file_path = os.path.join(temp_folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                     os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    def stream_image(self, event):
        app = Flask(__name__)
        CORS(app)  # Enable CORS for all routes
        OUTPUT_FOLDER = 'D:/DeepFake Vtuber/talking-head-anime-3-demo'

        @app.route('/result_feed')
        def get_image():
            def generate():
                while True:
                    numpy_image = self.last_output_numpy_image
                    pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA')
                    img_byte_array = io.BytesIO()
                    pil_image.save(img_byte_array, format='PNG')
                    img_byte_array.seek(0)
                    yield (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + img_byte_array.getvalue() + b'\r\n')

            return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

        def start_flask_server():
            app.run(host='127.0.0.1', port=8192, threaded=True)

        # Start the Flask server in a separate thread
        self.flask_thread = threading.Thread(target=start_flask_server)
        self.flask_thread.daemon = True  # Set the thread as a daemon
        self.flask_thread.start()
        print("image streamed to http://127.0.0.1:8192/result_feed")

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

    def update_capture_panel(self, event: wx.Event): # This panel might be removed or simplified for MediaPipe
        # This panel was for iFacialMocap specific rotation values.
        # For MediaPipe, this detailed breakdown might not be shown in the UI by default,
        # or it would show the HEAD_BONE_X, Y, Z values from the self.ifacialmocap_pose dictionary.
        # Since create_capture_panel and its contents are commented out later, this function body can be minimal.
        pass

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

    def draw_nothing_yet_string(self, dc):
        dc.Clear()
        font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
        dc.SetFont(font)
        w, h = dc.GetTextExtent("Nothing yet!")
        dc.DrawText("Nothing yet!", (self.poser.get_image_size() - w) // 2, (self.poser.get_image_size() - h) // 2)

    def paint_result_image_panel(self, event: wx.Event):
        wx.BufferedPaintDC(self.result_image_panel, self.result_image_bitmap)

    def thread_ai_convert(self, pose):
        with torch.no_grad():
            self.last_torch_image = self.poser.pose(self.torch_source_image, pose)[0].float()

    def update_result_image_bitmap(self, event: Optional[wx.Event] = None):
        # Optimization: don't process if window not active and MediaPipe not running with debug window
        if not self.is_mediapipe_running and not wx.GetApp().IsActive(): # wx.GetApp() is safer
            if not (self.mediapipe_input and self.show_mediapipe_debug_window_checkbox.IsChecked() and self.is_mediapipe_running):
                return

        # If mediapipe is not running AND we are not showing the debug window, then we might want to clear or do nothing
        if not self.is_mediapipe_running and not (self.mediapipe_input and self.show_mediapipe_debug_window_checkbox.IsChecked()):
            if self.torch_source_image is None:
                self.clear_result_image_to_background()
            # If there is a source image but no tracking, the image should probably remain static,
            # so we might not need to do anything here if last_torch_image is already set.
            # However, to ensure it's cleared if it was previously animating:
            elif self.last_torch_image is not None: # If it was animating, clear it
                 self.clear_result_image_to_background() # Or show static source image if desired
            return

        raw_pose_data = self.read_mediapipe_pose() # Changed from read_ifacialmocap_pose

        current_pose = self.pose_converter.convert(raw_pose_data, self.poseIsPerfectsync)

        if self.last_pose is not None and self.last_pose == current_pose and self.torch_source_image is not None: # Added source image check
            if self.same_pose_count >= 1:
                self.same_pose_count = 2
                return
            else:
                self.same_pose_count = 1
        else:
            self.same_pose_count = 0
        self.last_pose = current_pose

        image_size = self.poser.get_image_size()
        if self.torch_source_image is None:
            self.clear_result_image_to_background() # Centralized clearing logic
            return

        pose = torch.tensor(current_pose, device=self.device, dtype=self.poser.get_dtype())


        thread1 = threading.Thread(target = self.thread_ai_convert, args = (pose, ))

        if self.last_torch_image is None:
            thread1.start()
            thread1.join()
            return

        torch_image = self.last_torch_image
        thread1.start()

        with torch.no_grad():
            output_image = convert_linear_to_srgb((torch_image + 1.0) / 2.0)

            background_choice = self.output_background_choice.GetSelection()
            if background_choice == 0:
                pass
            else:
                background = torch.zeros(4, output_image.shape[1], output_image.shape[2], device=self.device)
                background[3, :, :] = 1.0
                if background_choice == 1:
                    background[1, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)
                elif background_choice == 2:
                    background[2, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)
                elif background_choice == 3:
                    output_image = self.blend_with_background(output_image, background)
                else:
                    background[0:3, :, :] = 1.0
                    output_image = self.blend_with_background(output_image, background)

            c, h, w = output_image.shape
            output_image = 255.0 * torch.transpose(output_image.reshape(c, h * w), 0, 1).reshape(h, w, c)
            output_image = output_image.byte()

        numpy_image = output_image.detach().cpu().numpy
        numpy_image = output_image.detach().cpu().numpy()
        self.last_output_numpy_image = numpy_image
        wx_image = wx.ImageFromBuffer(numpy_image.shape[0],
                                      numpy_image.shape[1],
                                      numpy_image[:, :, 0:3].tobytes(),
                                      numpy_image[:, :, 3].tobytes())
        wx_bitmap = wx_image.ConvertToBitmap()

        dc = wx.MemoryDC()
        dc.SelectObject(self.result_image_bitmap)
        dc.Clear()
        dc.DrawBitmap(wx_bitmap,
                      (image_size - numpy_image.shape[0]) // 2,
                      (image_size - numpy_image.shape[1]) // 2, True)
        del dc

        time_now = time.time_ns()
        if self.last_update_time is not None:
            elapsed_time = time_now - self.last_update_time
            fps = 1.0 / (elapsed_time / 10**9)
            if self.torch_source_image is not None:
                self.fps_statistics.add_fps(fps)
            self.fps_text.SetLabelText("FPS = %0.2f" % self.fps_statistics.get_average_fps())
        self.last_update_time = time_now

        self.result_image_panel.Refresh()

        thread1.join()

    def blend_with_background(self, numpy_image, background):
        alpha = numpy_image[3:4, :, :]
        color = numpy_image[0:3, :, :]
        new_color = color * alpha + (1.0 - alpha) * background[0:3, :, :]
        return torch.cat([new_color, background[3:4, :, :]], dim=0)

    def load_image(self, event: wx.Event):
        dir_name = "data/images"
        file_dialog = wx.FileDialog(self, "Choose an image", dir_name, "", "*.png", wx.FD_OPEN)
        if file_dialog.ShowModal() == wx.ID_OK:
            image_file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
            try:
                pil_image = resize_PIL_image(
                    extract_PIL_image_from_filelike(image_file_name),
                    (self.poser.get_image_size(), self.poser.get_image_size()))
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
                    torch_image = extract_pytorch_image_from_PIL_image(pil_image)
                    self.source_image_list.Append(file_dialog.GetFilename(), [pil_image, wx_image, torch_image, image_file_name])
                    self.source_image_list.SetSelection(image_list_index)
                    self.wx_source_image = wx_image
                    if image_list_index == 0:
                        tip = wx.adv.RichToolTip("Click to preview image.\nDoubleClick to output animation.", "") # "Quick Guide"
                        tip.SetTimeout(10000, 0)
                        tip.ShowFor(self.source_image_list)
                    # self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image) \
                    #     .to(self.device).to(self.poser.get_dtype())
                self.update_source_image_bitmap()
            except Exception as e:
                print(e)
                message_dialog = wx.MessageDialog(self, "Could not load image " + image_file_name, "Poser", wx.OK)
                message_dialog.ShowModal()
                message_dialog.Destroy()
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
        self.torch_source_image = image_sets[2].to(self.device).to(self.poser.get_dtype())
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
            self.torch_source_image = image_sets[2].to(self.device).to(self.poser.get_dtype())
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
        self.torch_source_image = None
        self.last_torch_image = None
        self.last_pose = None

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
        model_preserve = config_json_dict.get("model", 0)
        timer_preserve = int(config_json_dict.get("timer", 20))
        port_preserve = int(config_json_dict.get("port", 49983))
    else:
        model_preserve = 0
        timer_preserve = 20
        port_preserve = 49983

    if model_preserve == 0:
        model_str = "standard_float"
    elif model_preserve == 1:
        model_str = "separable_float"
    elif model_preserve == 2:
        model_str = "standard_half"
    elif model_preserve == 3:
        model_str = "separable_half"
    else:
        model_str = "standard_float"

    parser = argparse.ArgumentParser(description='Control characters with movement captured by iFacialMocap.')
    parser.add_argument(
        '--model',
        type=str,
        required=False,
        default=model_str,
        choices=['standard_float', 'separable_float', 'standard_half', 'separable_half'],
        help='The model to use.')
    parser.add_argument(
        '--timer',
        type=int,
        required=False,
        default=timer_preserve,
        # choices=range(5, 2000),
        help='Animation cycle ; 5-2000[ms].')
    # parser.add_argument( # Port argument removed for MediaPipe version
    #     '--port',
    #     type=int,
    #     required=False,
    #     default=port_preserve,
    #     # choices=range(0, 65535),
    #     help='Network port number to recieve motioncapture ; default is 49983.')
    args = parser.parse_args()

    # torch.set_num_interop_threads(4)
    # torch.set_num_threads(4)
    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        poser = load_poser(args.model, device)
    except RuntimeError as e:
        print(e)
        sys.exit()

    from tha3.mocap.ifacialmocap_poser_converter_25 import create_ifacialmocap_pose_converter

    pose_converter = create_ifacialmocap_pose_converter()

# preload models
    default_mocap_pose = create_default_ifacialmocap_pose()
    default_pose = pose_converter.convert(default_mocap_pose, True)
    poser.pose(torch.zeros(4, 512, 512).to(device).to(poser.get_dtype()), torch.tensor(default_pose, device=device, dtype=poser.get_dtype()))[0].float()

    # port_number is not used for socket binding in MediaPipe version,
    # but MainFrame constructor still expects a mocap_port argument. We pass None or a placeholder.
    # port_number = getattr(args, 'port', port_preserve) # Safely get port if it were still an arg
    # print(f"Note: Port argument is not used for UDP binding in MediaPipe version.")


    app = wx.App(False) # Added False for clean exit on some systems
    # Pass mocap_port as None or a default, it won't be used for socket binding by MediaPipe version
    main_frame = MainFrame(poser, pose_converter, device, mocap_port=None)
    main_frame.Show(True)

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
            if args.model == "standard_float":
                model_save = 0
            elif args.model == "separable_float":
                model_save = 1
            elif args.model == "standard_half":
                model_save = 2
            elif args.model == "separable_half":
                model_save = 3
            else:
                model_save = 0

            config_json_dict["model"] = model_save
            config_json_dict["timer"] = maintimer
            # config_json_dict["port"] = port_number # Port not saved for MediaPipe version
            with open('tha3sw_config.json', 'w') as f:
                json.dump(config_json_dict, f, indent=4)
        except Exception as e:
            print (e)

    # Animation timer will be started by on_start_capture when webcam starts, not here.
    # main_frame.animation_timer.Start(maintimer)
    app.MainLoop()

    def clear_result_image_to_background(self):
        image_size = self.poser.get_image_size()
        # Ensure background tensor is created on the correct device and dtype
        background = torch.zeros(4, image_size, image_size, device=self.device, dtype=self.poser.get_dtype())
        background_choice = self.output_background_choice.GetSelection()

        if background_choice == 0: # Transparent
            # Alpha is already 0 from torch.zeros. Color channels also 0.
            pass
        else:
            background[3, :, :] = 1.0 # Full alpha for solid backgrounds
            if background_choice == 1: # Green
                background[1, :, :] = 1.0
            elif background_choice == 2: # Blue
                background[2, :, :] = 1.0
            elif background_choice == 3: # Black
                # Color channels are already 0
                pass
            else: # White (choice 4)
                background[0:3, :, :] = 1.0 # R, G, B to 1.0 for white

        # The background tensor is already in the range [0,1] and correct dtype/device.
        # No need for (x+1)/2 conversion typically used for model outputs that are in [-1,1]
        output_image_srgb = background
        if self.poser.get_dtype() == torch.half: # Ensure it's float for byte conversion
            output_image_srgb = output_image_srgb.float()

        # Convert to displayable format (HWC, byte)
        # Permute from CHW to HWC for numpy/PIL
        output_image_hwc_float = output_image_srgb.permute(1, 2, 0)
        output_image_bytes = (255.0 * output_image_hwc_float).byte()

        numpy_image = output_image_bytes.detach().cpu().numpy()

        # Ensure we have an alpha channel for wx.ImageFromBuffer
        if numpy_image.shape[2] == 3: # If only RGB, add an alpha channel
            alpha_channel = np.full((numpy_image.shape[0], numpy_image.shape[1], 1), 255, dtype=np.uint8)
            if background_choice == 0: # Transparent
                 alpha_channel = np.full((numpy_image.shape[0], numpy_image.shape[1], 1), 0, dtype=np.uint8)
            numpy_image = np.concatenate((numpy_image, alpha_channel), axis=2)

        wx_image = wx.ImageFromBuffer(numpy_image.shape[1], # width
                                      numpy_image.shape[0], # height
                                      numpy_image[:, :, 0:3].tobytes(), # RGB data
                                      numpy_image[:, :, 3].tobytes())   # Alpha data
        wx_bitmap = wx_image.ConvertToBitmap()

        dc = wx.MemoryDC()
        dc.SelectObject(self.result_image_bitmap)
        dc.Clear() # Important to clear before drawing new bitmap
        dc.DrawBitmap(wx_bitmap,
                      (image_size - wx_bitmap.GetWidth()) // 2,
                      (image_size - wx_bitmap.GetHeight()) // 2, True)
        del dc
        self.result_image_panel.Refresh()
        self.last_output_numpy_image = numpy_image # Store this default view
