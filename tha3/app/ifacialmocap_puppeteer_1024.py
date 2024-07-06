import argparse
import os
import socket
import sys
import threading
import time
from typing import Optional

sys.path.append(os.getcwd())

from tha3.mocap.ifacialmocap_pose import create_default_ifacialmocap_pose
from tha3.mocap.ifacialmocap_v2 import IFACIALMOCAP_PORT, IFACIALMOCAP_START_STRING, parse_ifacialmocap_v2_pose, \
    parse_ifacialmocap_v1_pose, parse_meowface_pose, parse_vmc_pose, parse_vts_pose, parse_vmc_pose_list, parse_vmc_perfectsync_pose_list

from tha3.poser.modes.load_poser import load_poser

from tha3.mocap.ifacialmocap_poser_converter_25 import SmartPhoneApp

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
        super().__init__(None, wx.ID_ANY, "Face Motion Capture Puppeteer (Talking Head Anime 3)")
        self.pose_converter = pose_converter
        self.poser = poser
        self.device = device

        self.mocap_port = mocap_port

        self.image_size = 1024

        self.ifacialmocap_pose = create_default_ifacialmocap_pose()
        self.source_image_bitmap = wx.Bitmap(self.image_size, self.image_size)
        self.result_image_bitmap = wx.Bitmap(self.image_size, self.image_size)
        self.wx_source_image = None
        # self.torch_source_image = None
        self.last_pose = None
        self.fps_statistics = FpsStatistics()
        self.last_update_time = None
        self.same_pose_count = 0
        self.algorithm_mode = 0
        self.torch_source_sets = None

        self.torch_source_image = [None] * 4
        # self.last_torch_image = torch.zeros(4, 512, 512).float()
        self.last_torch_image = [None] * 4
        self.source_image_string = None

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
            composition_algo = config_json_dict.get("composition_algo", 0)
            capture_ip = config_json_dict.get("capture_ip", "192.168.0.1")
            mocap_method = config_json_dict.get("mocap_method", 2)
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

            image_list = config_json_dict.get("image_list_1024", [])
            image_select = config_json_dict.get("image_select_1024", -1)
            image_output = config_json_dict.get("image_output_1024", -1)

            self.algorithm_mode_choice.SetSelection(composition_algo)
            self.algorithm_mode = composition_algo

            self.capture_device_ip_text_ctrl.SetValue(capture_ip)

            self.pose_converter.sp_app_choice.SetSelection(mocap_method)
            self.pose_converter.change_sp_app(self)
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
                            (self.image_size, self.image_size))
                        w, h = pil_image.size
                        if pil_image.mode != 'RGBA':
                            raise Exception("Image must have alpha channel!")
                        else:
                            wx_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                            torch_image_temp = extract_pytorch_image_from_PIL_image(pil_image)
                    except Exception as e:
                        print(e)
                        image_name = image_name + "  (Image Loading Error!)"
                        w, h = self.image_size, self.image_size
                        pil_image = PIL.Image.new("RGBA", (w, h), (0, 0, 0, 0))
                        wx_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                        torch_image_temp = extract_pytorch_image_from_PIL_image(pil_image)
                    torch_image_temp_1 = torch_image_temp[:, ::2, ::2]
                    torch_image_temp_2 = torch_image_temp[:, 1::2, ::2]
                    torch_image_temp_3 = torch_image_temp[:, ::2, 1::2]
                    torch_image_temp_4 = torch_image_temp[:, 1::2, 1::2]
                    torch_image = [None] * 4
                    torch_image[0] = torch_image_temp_1
                    torch_image[1] = torch_image_temp_2
                    torch_image[2] = torch_image_temp_3
                    torch_image[3] = torch_image_temp_4
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
                    self.torch_source_sets = image_sets[2]
                    self.create_divided_images()
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
                composition_algo = self.algorithm_mode
                capture_ip = self.capture_device_ip_text_ctrl.GetValue()
                mocap_method = self.pose_converter.sp_app_choice.GetSelection()
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

                config_json_dict["composition_algo"] = composition_algo
                config_json_dict["capture_ip"] = capture_ip
                config_json_dict["mocap_method"] = mocap_method
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

                config_json_dict["image_list_1024"] = image_list
                config_json_dict["image_select_1024"] = image_select
                config_json_dict["image_output_1024"] = image_output

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
            return
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
        # print(VTS_START_STRING)
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

        image_size = self.poser.get_image_size()
        # image_size = self.image_size

        if True:
            self.input_panel = wx.Panel(self.animation_panel, size=(image_size, self.image_size + 0),
                                        style=wx.SIMPLE_BORDER)
            self.input_panel_sizer = wx.BoxSizer(wx.VERTICAL)
            self.input_panel.SetSizer(self.input_panel_sizer)
            self.input_panel.SetAutoLayout(1)
            self.animation_panel_sizer.Add(self.input_panel, 0, wx.FIXED_MINSIZE)

            self.create_connection_panel(self.input_panel)
            self.create_algorithm_panel(self.input_panel)
            self.create_reset_panel(self.input_panel)

            self.source_image_panel = wx.Panel(self.input_panel, size=(image_size, image_size), style=wx.SIMPLE_BORDER)
            self.source_image_panel.Bind(wx.EVT_PAINT, self.paint_source_image_panel)
            self.source_image_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
            self.input_panel_sizer.Add(self.source_image_panel, 0, wx.FIXED_MINSIZE)

 #Listbox for select and switch images
            image_list_text = wx.StaticText(self.input_panel, label="--- List of Image Files ---",
                                            style=wx.ALIGN_CENTER)
            self.input_panel_sizer.Add(image_list_text, 0, wx.EXPAND)

            self.source_image_list = wx.ListBox(self.input_panel, size=(image_size, 160), style=wx.LB_NEEDED_SB)
            self.source_image_list.Bind(wx.EVT_LISTBOX, self.source_image_select)
            self.source_image_list.Bind(wx.EVT_LISTBOX_DCLICK, self.source_image_apply)
            self.source_image_list.Bind(wx.EVT_KEY_UP, self.source_image_press_enter)
            self.input_panel_sizer.Add(self.source_image_list, 1, wx.EXPAND)

            self.load_image_button = wx.Button(self.input_panel, wx.ID_ANY, " \nLoad Image\n ")
            self.input_panel_sizer.Add(self.load_image_button, 0, wx.EXPAND)
            self.load_image_button.Bind(wx.EVT_BUTTON, self.load_image)

            separator0 = wx.StaticLine(self.input_panel, -1, size=(256, 4))
            self.input_panel_sizer.Add(separator0, flag = wx.GROW | wx.TOP, border = 12)

            background_text = wx.StaticText(self.input_panel, label="--- Background ---",
                                            style=wx.ALIGN_CENTER)
            self.input_panel_sizer.Add(background_text, 0, wx.EXPAND)

            self.output_background_choice = wx.Choice(
                self.input_panel,
                choices=[
                    "TRANSPARENT",
                    "GREEN",
                    "BLUE",
                    "BLACK",
                    "WHITE"
                ])
            self.output_background_choice.SetSelection(0)
            self.output_background_choice.Bind(wx.EVT_CHOICE, self.background_changed)
            self.input_panel_sizer.Add(self.output_background_choice, 0, wx.EXPAND)

            separator = wx.StaticLine(self.input_panel, -1, size=(256, 5))
            self.input_panel_sizer.Add(separator, 0, wx.EXPAND)

            self.fps_text = wx.StaticText(self.input_panel, label="")
            self.input_panel_sizer.Add(self.fps_text, wx.SizerFlags().Border())

            separator_bottom = wx.StaticLine(self.input_panel, -1, size=(256, 4))
            self.input_panel_sizer.Add(separator_bottom, flag = wx.GROW | wx.BOTTOM, border = 44)

            self.input_panel_sizer.Fit(self.input_panel)

        if True:
            self.pose_converter.init_pose_converter_panel(self.animation_panel)

        if True:
            self.animation_left_panel = wx.Panel(self.animation_panel, style=wx.SIMPLE_BORDER)
            self.animation_left_panel_sizer = wx.BoxSizer(wx.VERTICAL)
            self.animation_left_panel.SetSizer(self.animation_left_panel_sizer)
            self.animation_left_panel.SetAutoLayout(1)
            self.animation_panel_sizer.Add(self.animation_left_panel, 0, wx.EXPAND)

            self.result_image_panel = wx.Panel(self.animation_left_panel, size=(self.image_size, self.image_size),
                                               style=wx.SIMPLE_BORDER)
            self.result_image_panel.Bind(wx.EVT_PAINT, self.paint_result_image_panel)
            self.result_image_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
            self.animation_left_panel_sizer.Add(self.result_image_panel, 0, wx.FIXED_MINSIZE)

            self.animation_left_panel_sizer.Fit(self.animation_left_panel)

        self.animation_panel_sizer.Fit(self.animation_panel)

    def create_ui(self):
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.main_sizer)
        self.SetAutoLayout(1)

        self.capture_pose_lock = threading.Lock()

#        self.create_connection_panel(self)
#        self.main_sizer.Add(self.connection_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

        self.create_animation_panel(self)
        self.main_sizer.Add(self.animation_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 2))

#        self.create_capture_panel(self)
#        self.main_sizer.Add(self.capture_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

        self.main_sizer.Fit(self)

    def create_algorithm_panel(self, parent):
        self.algorithm_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.algorithm_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.algorithm_panel.SetSizer(self.algorithm_panel_sizer)
        self.algorithm_panel.SetAutoLayout(1)
        parent.GetSizer().Add(self.algorithm_panel, 0, wx.EXPAND)

        algorithm_mode_text = wx.StaticText(self.algorithm_panel, label=" Composition Algorithm  ",
                                               style=wx.ALIGN_CENTER)
        self.algorithm_panel_sizer.Add(algorithm_mode_text, 0, wx.EXPAND)

        self.algorithm_mode_choice = wx.Choice(
            self.algorithm_panel,
            choices=[
                "0 : Sharp, but Noisy",
                "1 : Midrange (Sharp)",
                "2 : Midrange (Soft)",
                "3 : Soft",
            ])
        self.algorithm_mode_choice.SetSelection(0)
        self.algorithm_panel_sizer.Add(self.algorithm_mode_choice, 1, wx.EXPAND)
        self.algorithm_mode_choice.Bind(wx.EVT_CHOICE, self.change_composition_algorithm)

        self.algorithm_panel_sizer.Fit(self.algorithm_panel)

    def create_reset_panel(self, parent):
        self.reset_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.reset_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.reset_panel.SetSizer(self.reset_panel_sizer)
        self.reset_panel.SetAutoLayout(1)
        parent.GetSizer().Add(self.reset_panel, 0, wx.EXPAND)

#        Add reset button
        self.reset_button = wx.Button(self.reset_panel, label="CLEAR Images")
        self.reset_panel_sizer.Add(self.reset_button, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.reset_button.Bind(wx.EVT_BUTTON, self.reset_clicked)

        space_text = wx.StaticText(self.reset_panel, label="  ", style=wx.ALIGN_RIGHT)
        self.reset_panel_sizer.Add(space_text, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))

        self.reset_ok_button = wx.Button(self.reset_panel, label="  OK  ")
        self.reset_panel_sizer.Add(self.reset_ok_button, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.reset_ok_button.Bind(wx.EVT_BUTTON, self.reset_ok_clicked)

        self.reset_cancel_button = wx.Button(self.reset_panel, label="CANCEL")
        self.reset_panel_sizer.Add(self.reset_cancel_button, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.reset_cancel_button.Bind(wx.EVT_BUTTON, self.reset_cancel_clicked)

        reset_spacer_line = wx.StaticLine(self.reset_panel, style = wx.LI_VERTICAL, size = (5, 10))
        self.reset_panel_sizer.Add(reset_spacer_line, flag = wx.GROW | wx.RIGHT, border = 150)

        self.reset_ok_button.Disable()
        self.reset_cancel_button.Disable()
        self.reset_panel_sizer.Fit(self.reset_panel)

    def create_connection_panel(self, parent):
        self.connection_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.connection_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.connection_panel.SetSizer(self.connection_panel_sizer)
        self.connection_panel.SetAutoLayout(1)
        parent.GetSizer().Add(self.connection_panel, 0, wx.EXPAND)

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

        self.connection_panel_sizer.Fit(self.connection_panel)
        
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
            draw_image = draw_image.Scale(self.poser.get_image_size(), self.poser.get_image_size(), wx.IMAGE_QUALITY_BOX_AVERAGE)
            draw_wx_image = wx.Image.ConvertToBitmap(draw_image)
            dc.Clear()
            dc.DrawBitmap(draw_wx_image, 0, 0, True)
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

    def thread_ai_convert(self, pose, area):
        with torch.no_grad():
            self.last_torch_image[area] = self.poser.pose(self.torch_source_image[area], pose)[0].float()

    def update_result_image_bitmap(self, event: Optional[wx.Event] = None):
        ifacialmocap_pose = self.read_ifacialmocap_pose()
        current_pose = self.pose_converter.convert(ifacialmocap_pose, self.poseIsPerfectsync)
        if self.last_pose is not None and self.last_pose == current_pose:
            if self.same_pose_count >= 1:
                self.same_pose_count = 2
                return
            else:
                self.same_pose_count = 1
        else:
            self.same_pose_count = 0
        self.last_pose = current_pose

        if self.torch_source_image[0] is None:
            # dc = wx.MemoryDC()
            # dc.SelectObject(self.result_image_bitmap)
            # self.draw_nothing_yet_string(dc)
            # del dc

            background = torch.zeros(4, self.image_size, self.image_size)
            background_choice = self.output_background_choice.GetSelection()
            if background_choice == 0:
                pass
            else:
                background[3, :, :] = 1.0
                if background_choice == 1:
                    background[1, :, :] = 1.0
                elif background_choice == 2:
                    background[2, :, :] = 1.0
                elif background_choice == 3:
                    pass
                else:
                    background[0:3, :, :] = 1.0
            output_image = background
            c, h, w = output_image.shape
            output_image = 255.0 * torch.transpose(output_image.reshape(c, h * w), 0, 1).reshape(h, w, c)
            output_image = output_image.byte()
            numpy_image = output_image.detach().cpu().numpy()
            # print(numpy_image[0,0])
            wx_image = wx.ImageFromBuffer(numpy_image.shape[0],
                                          numpy_image.shape[1],
                                          numpy_image[:, :, 0:3].tobytes(),
                                          numpy_image[:, :, 3].tobytes())
            wx_bitmap = wx_image.ConvertToBitmap()

            dc = wx.MemoryDC()
            dc.SelectObject(self.result_image_bitmap)
            dc.Clear()
            dc.DrawBitmap(wx_bitmap,
                          (self.image_size - numpy_image.shape[0]) // 2,
                          (self.image_size - numpy_image.shape[1]) // 2, True)
            del dc
            self.result_image_panel.Refresh()
            return

        pose = torch.tensor(current_pose, device=self.device, dtype=self.poser.get_dtype())

        thread0 = threading.Thread(target = self.thread_ai_convert, args = (pose, 0 ))
        thread1 = threading.Thread(target = self.thread_ai_convert, args = (pose, 1 ))
        thread2 = threading.Thread(target = self.thread_ai_convert, args = (pose, 2 ))
        thread3 = threading.Thread(target = self.thread_ai_convert, args = (pose, 3 ))

        if self.last_torch_image[3] is None:
            thread0.start()
            thread1.start()
            thread2.start()
            thread3.start()

            thread0.join()
            thread1.join()
            thread2.join()
            thread3.join()
            return

        torch_image  = torch.zeros(4, 1024, 1024).float().to(device)
        algo = self.algorithm_mode
        if algo == 0:
            torch_image[:, ::2, ::2]   = self.last_torch_image[0].to(device)
            torch_image[:, 1::2, ::2]  = self.last_torch_image[1].to(device)
            torch_image[:, ::2, 1::2]  = self.last_torch_image[2].to(device)
            torch_image[:, 1::2, 1::2] = self.last_torch_image[3].to(device)
            pass
        elif algo == 1:
            torch_image[:, ::2, ::2]   = self.last_torch_image[0].to(device)
            torch_image[:, 1::2, ::2]  = self.last_torch_image[1].to(device)
            torch_image[:, ::2, 1::2]  = self.last_torch_image[2].to(device)
            torch_image[:, 1::2, 1::2] = self.last_torch_image[3].to(device)
            pass
        elif algo == 2:
            torch_image[:, ::2, ::2]   = self.last_torch_image[0].to(device)
            torch_image[:, 1::2, ::2]  = self.last_torch_image[1].to(device)
            torch_image[:, ::2, 1::2]  = self.last_torch_image[2].to(device)
            torch_image[:, 1::2, 1::2] = self.last_torch_image[3].to(device)
            pass
        else:
            torch_image[:, ::2, ::2]   = self.last_torch_image[0].to(device)
            torch_image[:, 1::2, ::2]  = self.last_torch_image[0].to(device)
            torch_image[:, ::2, 1::2]  = self.last_torch_image[0].to(device)
            torch_image[:, 1::2, 1::2] = self.last_torch_image[0].to(device)
            torch_image[:, 1::2, ::2]  = torch_image[:, 1::2, ::2] + self.last_torch_image[1].to(device)
            torch_image[:, 2::2, ::2]  = torch_image[:, 2::2, ::2] + self.last_torch_image[1][:, 0:511, :].to(device)
            torch_image[:, 1::2, 1::2] = torch_image[:, 1::2, 1::2] + self.last_torch_image[1].to(device)
            torch_image[:, 2::2, 1::2] = torch_image[:, 2::2, 1::2] + self.last_torch_image[1][:, 0:511, :].to(device)
            torch_image[:, ::2, 1::2]  = torch_image[:, ::2, 1::2] + self.last_torch_image[2].to(device)
            torch_image[:, 1::2, 1::2] = torch_image[:, 1::2, 1::2] + self.last_torch_image[2].to(device)
            torch_image[:, ::2, 2::2]  = torch_image[:, ::2, 2::2] + self.last_torch_image[2][:, :, 0:511].to(device)
            torch_image[:, 1::2, 2::2] = torch_image[:, 1::2, 2::2] + self.last_torch_image[2][:, :, 0:511].to(device)
            torch_image[:, 1::2, 1::2] = torch_image[:, 1::2, 1::2] + self.last_torch_image[3].to(device)
            torch_image[:, 2::2, 1::2] = torch_image[:, 2::2, 1::2] + self.last_torch_image[3][:, 0:511, :].to(device)
            torch_image[:, 1::2, 2::2] = torch_image[:, 1::2, 2::2] + self.last_torch_image[3][:, :, 0:511].to(device)
            torch_image[:, 2::2, 2::2] = torch_image[:, 2::2, 2::2] + self.last_torch_image[3][:, 0:511, 0:511].to(device)
            torch_image[:, :, 0:1]     = torch_image[:, :, 0:1] * 2.0
            torch_image[:, 0:1, :]     = torch_image[:, 0:1, :] * 2.0
            torch_image = torch_image / 4.0
            pass

        thread0.start()
        thread1.start()
        thread2.start()
        thread3.start()

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

        numpy_image = output_image.detach().cpu().numpy()
        wx_image = wx.ImageFromBuffer(numpy_image.shape[0],
                                      numpy_image.shape[1],
                                      numpy_image[:, :, 0:3].tobytes(),
                                      numpy_image[:, :, 3].tobytes())
        wx_bitmap = wx_image.ConvertToBitmap()

        dc = wx.MemoryDC()
        dc.SelectObject(self.result_image_bitmap)
        dc.Clear()
        dc.DrawBitmap(wx_bitmap,
                      (self.image_size - numpy_image.shape[0]) // 2,
                      (self.image_size - numpy_image.shape[1]) // 2, True)
        del dc

        time_now = time.time_ns()
        if self.last_update_time is not None:
            elapsed_time = time_now - self.last_update_time
            fps = 1.0 / (elapsed_time / 10**9)
            if self.torch_source_image[0] is not None:
                self.fps_statistics.add_fps(fps)
            self.fps_text.SetLabelText("FPS = %0.2f" % self.fps_statistics.get_average_fps())
        self.last_update_time = time_now

        self.result_image_panel.Refresh()

        thread0.join()
        thread1.join()
        thread2.join()
        thread3.join()

    def blend_with_background(self, numpy_image, background):
        alpha = numpy_image[3:4, :, :]
        color = numpy_image[0:3, :, :]
        new_color = color * alpha + (1.0 - alpha) * background[0:3, :, :]
        return torch.cat([new_color, background[3:4, :, :]], dim=0)

    def change_composition_algorithm(self, event: wx.Event):
        self.algorithm_mode = self.algorithm_mode_choice.GetSelection()
        self.create_divided_images()
        pass

    def create_divided_images(self):
        if self.torch_source_sets is None:
            return

        algo = self.algorithm_mode
        if algo == 0:
            self.torch_source_image[0] = self.torch_source_sets[0].to(self.device).to(self.poser.get_dtype())
            self.torch_source_image[1] = self.torch_source_sets[1].to(self.device).to(self.poser.get_dtype())
            self.torch_source_image[2] = self.torch_source_sets[2].to(self.device).to(self.poser.get_dtype())
            self.torch_source_image[3] = self.torch_source_sets[3].to(self.device).to(self.poser.get_dtype())
            pass
        elif algo == 1:
            self.torch_source_image[0] = self.torch_source_sets[0] * 5.0 + self.torch_source_sets[1] + self.torch_source_sets[2] + self.torch_source_sets[3]
            self.torch_source_image[1] = self.torch_source_sets[0] + self.torch_source_sets[1] * 5.0 + self.torch_source_sets[2] + self.torch_source_sets[3]
            self.torch_source_image[2] = self.torch_source_sets[0] + self.torch_source_sets[1] + self.torch_source_sets[2] * 5.0 + self.torch_source_sets[3]
            self.torch_source_image[3] = self.torch_source_sets[0] + self.torch_source_sets[1] + self.torch_source_sets[2] + self.torch_source_sets[3] * 5.0
            self.torch_source_image[0] = self.torch_source_image[0] / 8.0
            self.torch_source_image[1] = self.torch_source_image[1] / 8.0
            self.torch_source_image[2] = self.torch_source_image[2] / 8.0
            self.torch_source_image[3] = self.torch_source_image[3] / 8.0
            self.torch_source_image[0] = self.torch_source_image[0].to(self.device).to(self.poser.get_dtype())
            self.torch_source_image[1] = self.torch_source_image[1].to(self.device).to(self.poser.get_dtype())
            self.torch_source_image[2] = self.torch_source_image[2].to(self.device).to(self.poser.get_dtype())
            self.torch_source_image[3] = self.torch_source_image[3].to(self.device).to(self.poser.get_dtype())
            pass
        elif algo == 2:
            torch_source_cp = torch.zeros(4, 1024, 1024).float()
            torch_source_cp[:, ::2, ::2] = self.torch_source_sets[0]
            torch_source_cp[:, 1::2, ::2] = self.torch_source_sets[1]
            torch_source_cp[:, ::2, 1::2] = self.torch_source_sets[2]
            torch_source_cp[:, 1::2, 1::2] = self.torch_source_sets[3]

            torch_source_temp = torch.zeros(4, 1026, 1026).float()
            torch_source_temp[:, 1:1025, 1:1025] = torch_source_cp
            torch_source_temp[:, 0:1, 1:1025] = torch_source_cp[:, 0:1, :]
            torch_source_temp[:, 1:1025, 0:1] = torch_source_cp[:, :, 0:1]
            torch_source_temp[:, 1025:1026, 1:1025] = torch_source_cp[:, 1023:1024, :]
            torch_source_temp[:, 1:1025, 1025:1026] = torch_source_cp[:, :, 1023:1024]
            torch_source_temp[:, 0:1, 0:1] = torch_source_cp[:, 0:1, 0:1]
            torch_source_temp[:, 1025:1026, 0:1] = torch_source_cp[:, 1023:1024, 0:1]
            torch_source_temp[:, 0:1, 1025:1026] = torch_source_cp[:, 0:1, 1023:1024]
            torch_source_temp[:, 1025:1026, 1025:1026] = torch_source_cp[:, 1023:1024, 1023:1024]

            self.torch_source_image[0] =   torch_source_temp[:, 0:1023:2, 0:1023:2]       + torch_source_temp[:, 1:1024:2, 0:1023:2] * 2.0 + torch_source_temp[:, 2:1025:2, 0:1023:2] \
                                                 + torch_source_temp[:, 0:1023:2, 1:1024:2] * 2.0 + torch_source_temp[:, 1:1024:2, 1:1024:2] * 4.0 + torch_source_temp[:, 2:1025:2, 1:1024:2] * 2.0 \
                                                 + torch_source_temp[:, 0:1023:2, 2:1025:2]       + torch_source_temp[:, 1:1024:2, 2:1025:2] * 2.0 + torch_source_temp[:, 2:1025:2, 2:1025:2]
            self.torch_source_image[1] =   torch_source_temp[:, 1:1024:2, 0:1023:2]       + torch_source_temp[:, 2:1025:2, 0:1023:2] * 2.0 + torch_source_temp[:, 3:1026:2, 0:1023:2] \
                                                 + torch_source_temp[:, 1:1024:2, 1:1024:2] * 2.0 + torch_source_temp[:, 2:1025:2, 1:1024:2] * 4.0 + torch_source_temp[:, 3:1026:2, 1:1024:2] * 2.0 \
                                                 + torch_source_temp[:, 1:1024:2, 2:1025:2]       + torch_source_temp[:, 2:1025:2, 2:1025:2] * 2.0 + torch_source_temp[:, 3:1026:2, 2:1025:2]
            self.torch_source_image[2] =   torch_source_temp[:, 0:1023:2, 1:1024:2]       + torch_source_temp[:, 1:1024:2, 1:1024:2] * 2.0 + torch_source_temp[:, 2:1025:2, 1:1024:2] \
                                                 + torch_source_temp[:, 0:1023:2, 2:1025:2] * 2.0 + torch_source_temp[:, 1:1024:2, 2:1025:2] * 4.0 + torch_source_temp[:, 2:1025:2, 2:1025:2] * 2.0 \
                                                 + torch_source_temp[:, 0:1023:2, 3:1026:2]       + torch_source_temp[:, 1:1024:2, 3:1026:2] * 2.0 + torch_source_temp[:, 2:1025:2, 3:1026:2]
            self.torch_source_image[3] =   torch_source_temp[:, 1:1024:2, 1:1024:2]       + torch_source_temp[:, 2:1025:2, 1:1024:2] * 2.0 + torch_source_temp[:, 3:1026:2, 1:1024:2] \
                                                 + torch_source_temp[:, 1:1024:2, 2:1025:2] * 2.0 + torch_source_temp[:, 2:1025:2, 2:1025:2] * 4.0 + torch_source_temp[:, 3:1026:2, 2:1025:2] * 2.0 \
                                                 + torch_source_temp[:, 1:1024:2, 3:1026:2]       + torch_source_temp[:, 2:1025:2, 3:1026:2] * 2.0 + torch_source_temp[:, 3:1026:2, 3:1026:2]
            self.torch_source_image[0] = self.torch_source_image[0] / 16.0
            self.torch_source_image[1] = self.torch_source_image[1] / 16.0
            self.torch_source_image[2] = self.torch_source_image[2] / 16.0
            self.torch_source_image[3] = self.torch_source_image[3] / 16.0
            self.torch_source_image[0] = self.torch_source_image[0].to(self.device).to(self.poser.get_dtype())
            self.torch_source_image[1] = self.torch_source_image[1].to(self.device).to(self.poser.get_dtype())
            self.torch_source_image[2] = self.torch_source_image[2].to(self.device).to(self.poser.get_dtype())
            self.torch_source_image[3] = self.torch_source_image[3].to(self.device).to(self.poser.get_dtype())
            pass
        else:
            self.torch_source_image[0] = self.torch_source_sets[0].to(self.device).to(self.poser.get_dtype())
            self.torch_source_image[1] = self.torch_source_sets[1].to(self.device).to(self.poser.get_dtype())
            self.torch_source_image[2] = self.torch_source_sets[2].to(self.device).to(self.poser.get_dtype())
            self.torch_source_image[3] = self.torch_source_sets[3].to(self.device).to(self.poser.get_dtype())
            pass
        self.last_pose = None

        return

    def load_image(self, event: wx.Event):
        dir_name = "data/images"
        file_dialog = wx.FileDialog(self, "Choose an image", dir_name, "", "*.png", wx.FD_OPEN)
        if file_dialog.ShowModal() == wx.ID_OK:
            image_file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
            try:
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
                    torch_image_temp = extract_pytorch_image_from_PIL_image(pil_image)
                    torch_image_temp_1 = torch_image_temp[:, ::2, ::2]
                    torch_image_temp_2 = torch_image_temp[:, 1::2, ::2]
                    torch_image_temp_3 = torch_image_temp[:, ::2, 1::2]
                    torch_image_temp_4 = torch_image_temp[:, 1::2, 1::2]
                    torch_image = [None] * 4
                    torch_image[0] = torch_image_temp_1
                    torch_image[1] = torch_image_temp_2
                    torch_image[2] = torch_image_temp_3
                    torch_image[3] = torch_image_temp_4

                    # print(torch_image)
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
        self.torch_source_sets = image_sets[2]
        self.create_divided_images()
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
            self.torch_source_sets = image_sets[2]
            self.create_divided_images()
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
        self.torch_source_image = [None] * 4
        self.last_torch_image = [None] * 4
        self.last_pose = None
        self.torch_source_sets = None

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
    parser.add_argument(
        '--port',
        type=int,
        required=False,
        default=port_preserve,
        # choices=range(0, 65535),
        help='Network port number to recieve motioncapture ; default is 49983.')
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

    port_number = args.port
    if port_number < 0:
        print("--port value is too small !!!")
        port_number = 49983
    elif port_number > 65535:
        print("--port value is too large !!!")
        port_number = 49983
    print(f"Receive motioncapture from port {port_number} .")

    app = wx.App()
    main_frame = MainFrame(poser, pose_converter, device, port_number)
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
            config_json_dict["port"] = port_number
            with open('tha3sw_config.json', 'w') as f:
                json.dump(config_json_dict, f, indent=4)
        except Exception as e:
            print (e)

    main_frame.animation_timer.Start(maintimer)
    app.MainLoop()
