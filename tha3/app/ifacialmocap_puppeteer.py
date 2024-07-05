import argparse
import os
import socket
import sys
import threading
import time
import PIL.Image
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
from tha3.mocap.ifacialmocap_v2 import IFACIALMOCAP_PORT, IFACIALMOCAP_START_STRING, parse_ifacialmocap_v2_pose, \
    parse_ifacialmocap_v1_pose, parse_meowface_pose, parse_vmc_pose
from tha3.poser.modes.load_poser import load_poser

from tha3.mocap.ifacialmocap_poser_converter_25 import SmartPhoneApp

import torch
import wx

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
    def __init__(self, poser: Poser, pose_converter: IFacialMocapPoseConverter, device:torch.device):
#        super().__init__(None, wx.ID_ANY, "iFacialMocap Puppeteer (Marigold)")
        super().__init__(None, wx.ID_ANY, "Face Motion Capture Puppeteer (Talking Head Anime 3)")
        self.pose_converter = pose_converter
        self.poser = poser
        self.device = device
        self.is_recording = False
        self.temp_folder = "temp"
        self.output_folder = "output"
        self.stream_folder = "stream"
        self.record_timer = None
        self.stream_timer = None
        self.record_counter = 0
        self.stream_counter = 0
        self.flask_thread = None

        self.ifacialmocap_pose = create_default_ifacialmocap_pose()
        self.source_image_bitmap = wx.Bitmap(self.poser.get_image_size(), self.poser.get_image_size())
        self.result_image_bitmap = wx.Bitmap(self.poser.get_image_size(), self.poser.get_image_size())
        self.wx_source_image = None
        self.torch_source_image = None
        self.last_pose = None
        self.fps_statistics = FpsStatistics()
        self.last_update_time = None
        self.same_pose_count = 0
        self.image_save_counter = 0
        self.last_output_numpy_image = None 

        # self.last_torch_image = torch.zeros(4, 512, 512).float()
        self.last_torch_image = None
        self.source_image_string = None

        self.create_receiving_socket()
        self.create_ui()
        self.create_timers()
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.update_source_image_bitmap()
        self.update_result_image_bitmap()

    def create_receiving_socket(self):
        self.receiving_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiving_socket.bind(("", IFACIALMOCAP_PORT))
        self.receiving_socket.setblocking(False)

    def create_timers(self):
        self.capture_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_capture_panel, id=self.capture_timer.GetId())
        self.animation_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_result_image_bitmap, id=self.animation_timer.GetId())

    def on_close(self, event: wx.Event):
        # Stop the timers
        self.animation_timer.Stop()
        self.capture_timer.Stop()

        # Close receiving socket
        self.receiving_socket.close()

        # Destroy the windows
        self.Destroy()
        event.Skip()

    def on_start_capture(self, event: wx.Event):
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

    def read_ifacialmocap_pose(self):
        if not self.animation_timer.IsRunning():
            return self.ifacialmocap_pose
        socket_bytes = None
        while True:
            try:
                socket_bytes = self.receiving_socket.recv(8192)
            except socket.error as e:
                break
        if socket_bytes is not None:
            socket_string = socket_bytes.decode("utf-8","ignore")
#            # For debug, please see the following string
#            pdb.set_trace()
#            print(socket_string)
            if ifadd.SP_APP_MODE == SmartPhoneApp.IPHONE:
#            SmartPhoneApp.IPHONE
                self.ifacialmocap_pose = parse_ifacialmocap_v2_pose(socket_string)
            elif ifadd.SP_APP_MODE == SmartPhoneApp.ANDROID:
#            SmartPhoneApp.ANDROID
                self.ifacialmocap_pose = parse_meowface_pose(socket_string)
            else:
                self.ifacialmocap_pose = parse_vmc_pose(socket_bytes)
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

        capture_device_ip_text = wx.StaticText(self.connection_panel, label="Capture Device IP:", style=wx.ALIGN_RIGHT)
        self.connection_panel_sizer.Add(capture_device_ip_text, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))

        self.capture_device_ip_text_ctrl = wx.TextCtrl(self.connection_panel, value="192.168.0.1")
        self.connection_panel_sizer.Add(self.capture_device_ip_text_ctrl, wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

        self.start_capture_button = wx.Button(self.connection_panel, label="START CAPTURE!")
        self.connection_panel_sizer.Add(self.start_capture_button, wx.SizerFlags(0).FixedMinSize().Border(wx.ALL, 3))
        self.start_capture_button.Bind(wx.EVT_BUTTON, self.on_start_capture)

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
        ifacialmocap_pose = self.read_ifacialmocap_pose()
        current_pose = self.pose_converter.convert(ifacialmocap_pose)
        

        if self.last_pose is not None and self.last_pose == current_pose:
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
            # dc = wx.MemoryDC()
            # dc.SelectObject(self.result_image_bitmap)
            # self.draw_nothing_yet_string(dc)
            # del dc

            background = torch.zeros(4, image_size, image_size)
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
            self.last_output_numpy_image = numpy_image
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
                          (image_size - numpy_image.shape[0]) // 2,
                          (image_size - numpy_image.shape[1]) // 2, True)
            del dc
            self.Refresh()
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
                self.result_image_panel.Bind(wx.EVT_ERASE_BACKGROUND, self.on_erase_background)
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

        self.Refresh()

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
                    self.source_image_string = "Image must have alpha channel!"
                else:
                    self.source_image_string = None
                    image_list_index = self.source_image_list.GetCount()
                    wx_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                    torch_image = extract_pytorch_image_from_PIL_image(pil_image)
                    self.source_image_list.Append(file_dialog.GetFilename(), [pil_image, wx_image, torch_image])
                    self.source_image_list.SetSelection(image_list_index)
                    self.wx_source_image = wx_image
                    # self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image) \
                    #     .to(self.device).to(self.poser.get_dtype())
                self.update_source_image_bitmap()
            except:
                message_dialog = wx.MessageDialog(self, "Could not load image " + image_file_name, "Poser", wx.OK)
                message_dialog.ShowModal()
                message_dialog.Destroy()
        file_dialog.Destroy()
        self.Refresh()

    def source_image_select(self, event: wx.Event):
        obj = event.GetEventObject()
        image_sets = obj.GetClientData(obj.GetSelection())
        self.source_image_string = None
        self.wx_source_image = image_sets[1]
        self.update_source_image_bitmap()
        self.Refresh()

    def source_image_apply(self, event: wx.Event):
        obj = event.GetEventObject()
        image_sets = obj.GetClientData(obj.GetSelection())
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
            image_sets = obj.GetClientData(obj.GetSelection())
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
    default_pose = pose_converter.convert(default_mocap_pose)
    poser.pose(torch.zeros(4, 512, 512).to(device).to(poser.get_dtype()), torch.tensor(default_pose, device=device, dtype=poser.get_dtype()))[0].float()

    app = wx.App()
    main_frame = MainFrame(poser, pose_converter, device)
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
