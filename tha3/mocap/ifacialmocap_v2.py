import math
import struct
import json
import copy

from tha3.mocap.ifacialmocap_constants import BLENDSHAPE_NAMES, HEAD_BONE_X, HEAD_BONE_Y, HEAD_BONE_Z, \
    RIGHT_EYE_BONE_X, RIGHT_EYE_BONE_Y, RIGHT_EYE_BONE_Z, LEFT_EYE_BONE_X, LEFT_EYE_BONE_Y, LEFT_EYE_BONE_Z, \
    HEAD_BONE_QUAT, LEFT_EYE_BONE_QUAT, RIGHT_EYE_BONE_QUAT, ROTATION_NAMES

from tha3.mocap.ifacialmocap_constants import CHEEK_SQUINT_LEFT, CHEEK_SQUINT_RIGHT

IFACIALMOCAP_PORT = 49983
IFACIALMOCAP_START_STRING = "iFacialMocap_sahuasouryya9218sauhuiayeta91555dy3719|sendDataVersion=v2".encode('utf-8')

# VTS_PORT = 21412
# VTS_START_STRING = '{"messageType":"iOSTrackingDataRequest","time":10.0,"sentBy":"THA3SW","ports":[11125]}'.encode('utf-8')

VMC_PARAM_NAMES = [
    "Joy", 
    "Angry", 
    "Sorrow", 
    "Fun", 
    "Surprised", 
    "A", 
    "I", 
    "U", 
    "E", 
    "O", 
    "Blink_L", 
    "Blink_R", 
    "Blink", 
    "LookUp",
    "LookDown",
    "LookLeft",
    "LookRight",
    "NEUTRAL",
    # HEAD_BONE_X, 
    # HEAD_BONE_Y, 
    # HEAD_BONE_Z, 
    # RIGHT_EYE_BONE_X, 
    # RIGHT_EYE_BONE_Y, 
    # RIGHT_EYE_BONE_Z, 
    # LEFT_EYE_BONE_X, 
    # LEFT_EYE_BONE_Y, 
    # LEFT_EYE_BONE_Z, 
]

VMC_PERFECTSYNC_PARAM_NAMES = [param.lower() for param in BLENDSHAPE_NAMES]

def parse_ifacialmocap_v2_pose(ifacialmocap_output):
    output = {}
    isDataCorrect = False
    parts = ifacialmocap_output.split("|")
    for part in parts:
        part = part.strip()
        if len(part) == 0:
            continue
        if "&" in part:
            components = part.split("&")
            if len(components) == 2:
                key = components[0]
                value = float(components[1]) / 100.0
                if key.endswith("_L"):
                    key = key[:-2] + "Left"
                elif key.endswith("_R"):
                    key = key[:-2] + "Right"
                if key in BLENDSHAPE_NAMES:
                    output[key] = value
            # else:
            #     raise AssertionError("Incorrect blendshape on ifacialmocap_v2")
        elif part.startswith("=head#"):
            components = part[len("=head#"):].split(",")
            assert len(components) == 6
            isDataCorrect = True
            output[HEAD_BONE_X] = float(components[0]) * math.pi / 180
            output[HEAD_BONE_Y] = float(components[1]) * math.pi / 180
            output[HEAD_BONE_Z] = float(components[2]) * math.pi / 180
        elif part.startswith("rightEye#"):
            components = part[len("rightEye#"):].split(",")
            output[RIGHT_EYE_BONE_X] = float(components[0]) * math.pi / 180
            output[RIGHT_EYE_BONE_Y] = float(components[1]) * math.pi / 180
            output[RIGHT_EYE_BONE_Z] = float(components[2]) * math.pi / 180
        elif part.startswith("leftEye#"):
            components = part[len("leftEye#"):].split(",")
            output[LEFT_EYE_BONE_X] = float(components[0]) * math.pi / 180
            output[LEFT_EYE_BONE_Y] = float(components[1]) * math.pi / 180
            output[LEFT_EYE_BONE_Z] = float(components[2]) * math.pi / 180
    for keyname in BLENDSHAPE_NAMES:
        if keyname not in output:
            output[keyname] = 0.0
    for rotate_bone_param in ROTATION_NAMES:
        if rotate_bone_param not in output:
            output[rotate_bone_param] = 0.0
    output[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    return (output, isDataCorrect)


def parse_ifacialmocap_v1_pose(ifacialmocap_output):
    output = {}
    isDataCorrect = False
    parts = ifacialmocap_output.split("|")
    for part in parts:
        part = part.strip()
        if len(part) == 0:
            continue
        if part.startswith("=head#"):
            components = part[len("=head#"):].split(",")
            assert len(components) == 6
            isDataCorrect = True
            output[HEAD_BONE_X] = float(components[0]) * math.pi / 180
            output[HEAD_BONE_Y] = float(components[1]) * math.pi / 180
            output[HEAD_BONE_Z] = float(components[2]) * math.pi / 180
        elif part.startswith("rightEye#"):
            components = part[len("rightEye#"):].split(",")
            output[RIGHT_EYE_BONE_X] = float(components[0]) * math.pi / 180
            output[RIGHT_EYE_BONE_Y] = float(components[1]) * math.pi / 180
            output[RIGHT_EYE_BONE_Z] = float(components[2]) * math.pi / 180
        elif part.startswith("leftEye#"):
            components = part[len("leftEye#"):].split(",")
            output[LEFT_EYE_BONE_X] = float(components[0]) * math.pi / 180
            output[LEFT_EYE_BONE_Y] = float(components[1]) * math.pi / 180
            output[LEFT_EYE_BONE_Z] = float(components[2]) * math.pi / 180
        else:
            components = part.split("-")
            if len(components) == 2:
                key = components[0]
                value = float(components[1]) / 100.0
                if key.endswith("_L"):
                    key = key[:-2] + "Left"
                elif key.endswith("_R"):
                    key = key[:-2] + "Right"
                if key in BLENDSHAPE_NAMES:
                    output[key] = value
            elif len(components) == 3:
                key = components[0]
                value = -float(components[2]) / 100.0
                if key.endswith("_L"):
                    key = key[:-2] + "Left"
                elif key.endswith("_R"):
                    key = key[:-2] + "Right"
                if key in BLENDSHAPE_NAMES:
                    output[key] = value
    for keyname in BLENDSHAPE_NAMES:
        if keyname not in output:
            output[keyname] = 0.0
    for rotate_bone_param in ROTATION_NAMES:
        if rotate_bone_param not in output:
            output[rotate_bone_param] = 0.0
    output[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    return (output, isDataCorrect)

def parse_meowface_pose(ifacialmocap_output):
    output = {}
    isDataCorrect = False
    
#    Setting default values (MeowFace does not have these values in texts)
    output[CHEEK_SQUINT_LEFT] = 0.0
    output[CHEEK_SQUINT_RIGHT] = 0.0

    parts = ifacialmocap_output.split("|")
    for part in parts:
        part = part.strip()

#        remove space character in MeowFace texts ' & '
        part = part.replace(' ', '')

        if len(part) == 0:
            continue
        if "&" in part:
            components = part.split("&")
            if len(components) == 2:
                key = components[0]
                value = float(components[1]) / 100.0
                if key.endswith("_L"):
                    key = key[:-2] + "Left"
                elif key.endswith("_R"):
                    key = key[:-2] + "Right"
                if key in BLENDSHAPE_NAMES:
                    output[key] = value
            # else:
            #     raise AssertionError("Incorrect blendshape on meowface")
        elif part.startswith("=head#"):
            components = part[len("=head#"):].split(",")
            assert len(components) == 6
            isDataCorrect = True
            output[HEAD_BONE_X] = float(components[0]) * math.pi / 180
            output[HEAD_BONE_Y] = float(components[1]) * math.pi / 180
            output[HEAD_BONE_Z] = float(components[2]) * math.pi / 180
        elif part.startswith("rightEye#"):
            components = part[len("rightEye#"):].split(",")
            output[RIGHT_EYE_BONE_X] = float(components[0]) * math.pi / 180
            output[RIGHT_EYE_BONE_Y] = float(components[1]) * math.pi / 180
            output[RIGHT_EYE_BONE_Z] = float(components[2]) * math.pi / 180
        elif part.startswith("leftEye#"):
            components = part[len("leftEye#"):].split(",")
            output[LEFT_EYE_BONE_X] = float(components[0]) * math.pi / 180
            output[LEFT_EYE_BONE_Y] = float(components[1]) * math.pi / 180
            output[LEFT_EYE_BONE_Z] = float(components[2]) * math.pi / 180
    for keyname in BLENDSHAPE_NAMES:
        if keyname not in output:
            output[keyname] = 0.0
    for rotate_bone_param in ROTATION_NAMES:
        if rotate_bone_param not in output:
            output[rotate_bone_param] = 0.0
    output[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    return (output, isDataCorrect)

def parse_vts_pose(ifacialmocap_output):
    output = {}
    dict_parts = json.loads(ifacialmocap_output)

# VTubeStudio (android app.) will not send these blendshape parameters.   
# CHEEK_SQUINT_LEFT = "cheekSquintLeft"
# CHEEK_SQUINT_RIGHT = "cheekSquintRight"

# JAW_FORWARD = "jawForward"
# MOUTH_SHRUG_LOWER = "mouthShrugLower"
# MOUTH_CLOSE = "mouthClose"
# MOUTH_DIMPLE_LEFT = "mouthDimpleLeft"
# MOUTH_PRESS_LEFT = "mouthPressLeft"
# MOUTH_STRETCH_LEFT = "mouthStretchLeft"
# MOUTH_DIMPLE_RIGHT = "mouthDimpleRight"
# MOUTH_PRESS_RIGHT = "mouthPressRight"
# MOUTH_STRETCH_RIGHT = "mouthStretchRight"

    browInnerUpLeft = 0.0
    browInnerUpRight = 0.0
    isDataCorrect = False

    isfacefound = dict_parts.get("FaceFound", None)
    if isfacefound is None:
        isDataCorrect = False
    else:
        isDataCorrect = True
        if isfacefound is not True:
            # output = copy.deepcopy(pose_cache)
            # return output
            isfacefound_bool = False
        else:
            isfacefound_bool = True

    dict_rotation = dict_parts.get("Rotation")
    output[HEAD_BONE_X] = float(dict_rotation.get("y", 0.0)) * math.pi / 180
    output[HEAD_BONE_Y] = float(dict_rotation.get("x", 0.0)) * math.pi / 180
    output[HEAD_BONE_Z] = float(dict_rotation.get("z", 0.0)) * math.pi / 180

    # x: <> , y: ^v
    dict_eyeright = dict_parts.get("EyeRight")
    output[RIGHT_EYE_BONE_X] = float(dict_eyeright.get("x", 0.0)) * math.pi / 180
    output[RIGHT_EYE_BONE_Y] = float(dict_eyeright.get("y", 0.0)) * math.pi / 180
    output[RIGHT_EYE_BONE_Z] = float(dict_eyeright.get("z", 0.0)) * math.pi / 180
    dict_eyeleft = dict_parts.get("EyeLeft")
    output[LEFT_EYE_BONE_X] = float(dict_eyeleft.get("x", 0.0)) * math.pi / 180
    output[LEFT_EYE_BONE_Y] = float(dict_eyeleft.get("y", 0.0)) * math.pi / 180
    output[LEFT_EYE_BONE_Z] = float(dict_eyeleft.get("z", 0.0)) * math.pi / 180

    list_blendshape = dict_parts.get("BlendShapes")
    for part in list_blendshape:
        if len(part) == 0:
            continue
        key = part.get("k")
        value = float(part.get("v"))
        key = key[0].lower() + key[1:]
        if key.endswith("_L"):
            key = key[:-2] + "Left"
        elif key.endswith("_R"):
            key = key[:-2] + "Right"
        if key in BLENDSHAPE_NAMES:
            output[key] = value
        elif key == "browInnerUpLeft":
            browInnerUpLeft = value
        elif key == "browInnerUpRight":
            browInnerUpRight = value

    if "browInnerUp" not in output:
        output["browInnerUp"] = (browInnerUpLeft + browInnerUpRight) / 2.0

    if "cheekSquintLeft" not in output:
        output["cheekSquintLeft"] = output.get("eyeSquintLeft", 0.0)
    if "cheekSquintRight" not in output:
        output["cheekSquintRight"] = output.get("eyeSquintRight", 0.0)
    for keyname in BLENDSHAPE_NAMES:
        if keyname not in output:
            output[keyname] = 0.0

    output[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    return (output, isDataCorrect, isfacefound_bool)

def parse_vmc_pose_list(vmc_list = [], pose_cache = {}):
    output = {}
    param_full = False
    for keydefault in BLENDSHAPE_NAMES:
        output[keydefault] = pose_cache.get(keydefault, 0.0)
    for ready_param in VMC_PARAM_NAMES:
        output[ready_param] = pose_cache.get(ready_param, 0.0)
    for rotate_bone_param in ROTATION_NAMES:
        output[rotate_bone_param] = pose_cache.get(rotate_bone_param, 0.0)
    output[HEAD_BONE_QUAT] = pose_cache.get(HEAD_BONE_QUAT, [0.0, 0.0, 0.0, 1.0])
    output[LEFT_EYE_BONE_QUAT] = pose_cache.get(LEFT_EYE_BONE_QUAT, [0.0, 0.0, 0.0, 1.0])
    output[RIGHT_EYE_BONE_QUAT] = pose_cache.get(RIGHT_EYE_BONE_QUAT, [0.0, 0.0, 0.0, 1.0])

    vmc_list_len = len(vmc_list)
    count = vmc_list_len
    param_flags = [False, False, False, False, False]
    # 0:BlendShapeApply 1:BlendShape 2:Head 3:RightEye 4:LeftEye
    isReadBlendshape = False
    isDataCorrect = False

    while count > 0:
        count -= 1
        vmc_output = vmc_list[count]

        parts = parse_osc(vmc_output)

        parts_count = len(parts)
        while parts_count > 0:
            parts_count -= 1
            part = parts[parts_count]
            # partstr = part.decode("utf-8","replace")
            vmcparts = part.split(b"\x00")
            vmcurl = vmcparts[0].decode("utf-8","ignore")
            if "/VMC" in vmcurl:
                isDataCorrect = True
            if "/Blend" in vmcurl:
                if "/Blend/Apply" in vmcurl:
                    if param_flags[0] == False:
                        param_flags[0] = True
                else:
                    if param_flags[0] == True and param_flags[1] == False:
                        isReadBlendshape = True
                        key = vmcparts[3].decode("utf-8","ignore")
                        value = struct.unpack('>f', part[part.__len__()-4:])[0]

                        # VRM1 -> VRM0 / upper lower diff
                        if True:
                            key_lower = key.lower()
                            if key_lower == ("happy"):
                                key = "Joy"
                            elif key_lower == ("angry"):
                                key = "Angry"
                            elif key_lower == ("sad"):
                                key = "Sorrow"
                            elif key_lower == ("relaxed"):
                                key = "Fun"
                            elif key_lower == ("surprised"):
                                key = "Surprised"
                            elif key_lower == ("aa"):
                                key = "A"
                            elif key_lower == ("ih"):
                                key = "I"
                            elif key_lower == ("ou"):
                                key = "U"
                            elif key_lower == ("ee"):
                                key = "E"
                            elif key_lower == ("oh"):
                                key = "O"
                            elif key_lower == ("blinkleft"):
                                key = "Blink_L"
                            elif key_lower == ("blinkright"):
                                key = "Blink_R"
                            elif key_lower == ("blink"):
                                key = "Blink"
                            elif key_lower == ("lookup"):
                                key = "LookUp"
                            elif key_lower == ("lookdown"):
                                key = "LookDown"
                            elif key_lower == ("lookleft"):
                                key = "LookLeft"
                            elif key_lower == ("lookright"):
                                key = "LookRight"
                            elif key_lower == ("neutral"):
                                key = "NEUTRAL"

                            elif key_lower == ("joy"):
                                key = "Joy"
                            elif key_lower == ("angry"):
                                key = "Angry"
                            elif key_lower == ("sorrow"):
                                key = "Sorrow"
                            elif key_lower == ("fun"):
                                key = "Fun"
                            elif key_lower == ("a"):
                                key = "A"
                            elif key_lower == ("i"):
                                key = "I"
                            elif key_lower == ("u"):
                                key = "U"
                            elif key_lower == ("e"):
                                key = "E"
                            elif key_lower == ("o"):
                                key = "O"
                            elif key_lower == ("blink_l"):
                                key = "Blink_L"
                            elif key_lower == ("blink_r"):
                                key = "Blink_R"

                        if key in VMC_PARAM_NAMES:
                            output[key] = value

                        # if key.endswith("_L"):
                        #     key = key[:-2] + "Left"
                        # elif key.endswith("_R"):
                        #     key = key[:-2] + "Right"
                        # if key in BLENDSHAPE_NAMES:
                        #     output[key] = value
            else:
                if param_flags[0] == True and param_flags[1] == False:
                    if isReadBlendshape == True:
                        param_flags[1] = True
                

            if "/Bone" in vmcurl:
                key = vmcparts[6].decode("utf-8","ignore").lower()
                values = struct.unpack('>fffffff', part[part.__len__()-28:])
                if key == ("head"):
                    if param_flags[2] == False:
                        param_flags[2] = True
                        headangles = quaternion_to_euler(values[3], values[4], values[5], values[6], "yxz")
                        output[HEAD_BONE_X] = headangles[0]
                        output[HEAD_BONE_Y] = -headangles[1]
                        output[HEAD_BONE_Z] = -headangles[2]
                        output[HEAD_BONE_QUAT] = [values[3], values[4], values[5], values[6]]
                elif key == ("righteye"):
                    if param_flags[3] == False:
                        param_flags[3] = True
                        righteyeangles = quaternion_to_euler(values[3], values[4], values[5], values[6], "yxz")
                        output[RIGHT_EYE_BONE_X] = -righteyeangles[0]
                        output[RIGHT_EYE_BONE_Y] = righteyeangles[1]
                        output[RIGHT_EYE_BONE_Z] = righteyeangles[2]
                        output[RIGHT_EYE_BONE_QUAT] = [values[3], values[4], values[5], values[6]]
                elif key == ("lefteye"):
                    if param_flags[4] == False:
                        param_flags[4] = True
                        lefteyeangles = quaternion_to_euler(values[3], values[4], values[5], values[6], "yxz")
                        output[LEFT_EYE_BONE_X] = -lefteyeangles[0]
                        output[LEFT_EYE_BONE_Y] = lefteyeangles[1]
                        output[LEFT_EYE_BONE_Z] = lefteyeangles[2]
                        output[LEFT_EYE_BONE_QUAT] = [values[3], values[4], values[5], values[6]]

        if param_flags == [True, True, True, True, True]:
            param_full = True
            break

    return (output, isDataCorrect, param_full)

def parse_vmc_perfectsync_pose_list(vmc_list = [], pose_cache = {}):
    output = {}
    param_full = False
    for keydefault in BLENDSHAPE_NAMES:
        output[keydefault] = pose_cache.get(keydefault, 0.0)
    for ready_param in VMC_PARAM_NAMES:
        output[ready_param] = pose_cache.get(ready_param, 0.0)
    for rotate_bone_param in ROTATION_NAMES:
        output[rotate_bone_param] = pose_cache.get(rotate_bone_param, 0.0)
    output[HEAD_BONE_QUAT] = pose_cache.get(HEAD_BONE_QUAT, [0.0, 0.0, 0.0, 1.0])
    output[LEFT_EYE_BONE_QUAT] = pose_cache.get(LEFT_EYE_BONE_QUAT, [0.0, 0.0, 0.0, 1.0])
    output[RIGHT_EYE_BONE_QUAT] = pose_cache.get(RIGHT_EYE_BONE_QUAT, [0.0, 0.0, 0.0, 1.0])

    vmc_list_len = len(vmc_list)
    count = vmc_list_len
    param_flags = [False, False, False, False, False]
    # 0:BlendShapeApply 1:BlendShape 2:Head 3:RightEye 4:LeftEye
    isReadBlendshape = False
    isDataCorrect = False

    while count > 0:
        count -= 1
        vmc_output = vmc_list[count]

        parts = parse_osc(vmc_output)

        parts_count = len(parts)
        while parts_count > 0:
            parts_count -= 1
            part = parts[parts_count]
            # partstr = part.decode("utf-8","replace")
            vmcparts = part.split(b"\x00")
            vmcurl = vmcparts[0].decode("utf-8","ignore")
            if "/VMC" in vmcurl:
                isDataCorrect = True
            if "/Blend" in vmcurl:
                if "/Blend/Apply" in vmcurl:
                    if param_flags[0] == False:
                        param_flags[0] = True
                else:
                    if param_flags[0] == True and param_flags[1] == False:
                        isReadBlendshape = True
                        key = vmcparts[3].decode("utf-8","ignore")
                        value = struct.unpack('>f', part[part.__len__()-4:])[0]

                        # VRM1 -> VRM0 / upper lower diff
                        if True:
                            key_lower = key.lower()
                        if key_lower.endswith("_l"):
                            key_lower = key_lower[:-2] + "left"
                        elif key_lower.endswith("_r"):
                            key_lower = key_lower[:-2] + "right"

                        if key_lower in VMC_PERFECTSYNC_PARAM_NAMES:
                            key_index = VMC_PERFECTSYNC_PARAM_NAMES.index(key_lower)
                            key_blendshape = BLENDSHAPE_NAMES[key_index]
                            output[key_blendshape] = value

            else:
                if param_flags[0] == True and param_flags[1] == False:
                    if isReadBlendshape == True:
                        param_flags[1] = True
                

            if "/Bone" in vmcurl:
                key = vmcparts[6].decode("utf-8","ignore").lower()
                values = struct.unpack('>fffffff', part[part.__len__()-28:])
                if key == ("head"):
                    if param_flags[2] == False:
                        param_flags[2] = True
                        headangles = quaternion_to_euler(values[3], values[4], values[5], values[6], "yxz")
                        output[HEAD_BONE_X] = headangles[0]
                        output[HEAD_BONE_Y] = -headangles[1]
                        output[HEAD_BONE_Z] = -headangles[2]
                        output[HEAD_BONE_QUAT] = [values[3], values[4], values[5], values[6]]
                elif key == ("righteye"):
                    if param_flags[3] == False:
                        param_flags[3] = True
                        righteyeangles = quaternion_to_euler(values[3], values[4], values[5], values[6], "yxz")
                        output[RIGHT_EYE_BONE_X] = -righteyeangles[0]
                        output[RIGHT_EYE_BONE_Y] = righteyeangles[1]
                        output[RIGHT_EYE_BONE_Z] = righteyeangles[2]
                        output[RIGHT_EYE_BONE_QUAT] = [values[3], values[4], values[5], values[6]]
                elif key == ("lefteye"):
                    if param_flags[4] == False:
                        param_flags[4] = True
                        lefteyeangles = quaternion_to_euler(values[3], values[4], values[5], values[6], "yxz")
                        output[LEFT_EYE_BONE_X] = -lefteyeangles[0]
                        output[LEFT_EYE_BONE_Y] = lefteyeangles[1]
                        output[LEFT_EYE_BONE_Z] = lefteyeangles[2]
                        output[LEFT_EYE_BONE_QUAT] = [values[3], values[4], values[5], values[6]]

        if param_flags[0:3] == [True, True, True]:
            param_full = True
            break

    return (output, isDataCorrect, param_full)

def parse_vmc_pose(vmc_output):
    output = {}

    for keydefault in BLENDSHAPE_NAMES:
        output[keydefault] = 0.0
    output["Joy"] = 0.0
    output["Angry"] = 0.0
    output["Sorrow"] = 0.0
    output["Fun"] = 0.0
    output["Surprised"] = 0.0
    output["A"] = 0.0
    output["I"] = 0.0
    output["U"] = 0.0
    output["E"] = 0.0
    output["O"] = 0.0
    output["BlinkLeft"] = 0.0
    output["BlinkRight"] = 0.0
    output["Blink"] = 0.0
    output[HEAD_BONE_X] = 0.0
    output[HEAD_BONE_Y] = 0.0
    output[HEAD_BONE_Z] = 0.0
    output[RIGHT_EYE_BONE_X] = 0.0
    output[RIGHT_EYE_BONE_Y] = 0.0
    output[RIGHT_EYE_BONE_Z] = 0.0
    output[LEFT_EYE_BONE_X] = 0.0
    output[LEFT_EYE_BONE_Y] = 0.0
    output[LEFT_EYE_BONE_Z] = 0.0
    output[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    parts = parse_osc(vmc_output)

    for part in parts:
        partstr = part.decode("utf-8","replace")
        vmcparts = part.split(b"\x00")
        vmcurl = vmcparts[0].decode("utf-8","ignore")
        if "/Blend" in vmcurl:
            key = vmcparts[3].decode("utf-8","ignore")
            value = struct.unpack('>f', part[part.__len__()-4:])[0]

            # VRM1 -> VRM0
            if key == ("happy"):
                key = "Joy"
            elif key == ("angry"):
                key = "Angry"
            elif key == ("sad"):
                key = "Sorrow"
            elif key == ("relaxed"):
                key = "Fun"
            elif key == ("aa"):
                key = "A"
            elif key == ("ih"):
                key = "I"
            elif key == ("ou"):
                key = "U"
            elif key == ("ee"):
                key = "E"
            elif key == ("oh"):
                key = "O"
            elif key == ("blinkLeft"):
                key = "Blink_L"
            elif key == ("blinkRight"):
                key = "Blink_R"

            if key.endswith("_L"):
                key = key[:-2] + "Left"
            elif key.endswith("_R"):
                key = key[:-2] + "Right"
             # if key in BLENDSHAPE_NAMES:
            #     output[key] = value
            output[key] = value
        elif "/Bone" in vmcurl:
            key = vmcparts[6].decode("utf-8","ignore")
            values = struct.unpack('>fffffff', part[part.__len__()-28:])
            if key == ("Head"):
                headangles = quaternion_to_euler(values[3], values[4], values[5], values[6], "yxz")
                output[HEAD_BONE_X] = headangles[0]
                output[HEAD_BONE_Y] = -headangles[1]
                output[HEAD_BONE_Z] = -headangles[2]
                output[HEAD_BONE_QUAT] = [values[3], values[4], values[5], values[6]]
            elif key == ("RightEye"):
                righteyeangles = quaternion_to_euler(values[3], values[4], values[5], values[6], "yxz")
                output[RIGHT_EYE_BONE_X] = -righteyeangles[0]
                output[RIGHT_EYE_BONE_Y] = righteyeangles[1]
                output[RIGHT_EYE_BONE_Z] = righteyeangles[2]
                output[LEFT_EYE_BONE_QUAT] = [values[3], values[4], values[5], values[6]]
            elif key == ("LeftEye"):
                lefteyeangles = quaternion_to_euler(values[3], values[4], values[5], values[6], "yxz")
                output[LEFT_EYE_BONE_X] = -lefteyeangles[0]
                output[LEFT_EYE_BONE_Y] = lefteyeangles[1]
                output[LEFT_EYE_BONE_Z] = lefteyeangles[2]
                output[RIGHT_EYE_BONE_QUAT] = [values[3], values[4], values[5], values[6]]
    # output[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    # output[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    # output[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    # print(output)
    return output

def quaternion_to_euler(x, y, z, w, order):
    eulerangle_xyz = [0.0, 0.0, 0.0]
    if order == ("xyz"):
        eulerangle_xyz[1] = math.asin(2.0 * x * z + 2.0 * y * w)
        eulerangle_xyz[0] = math.atan(-((2.0 * y * z - 2.0 * x * w) / (2.0 * w * w + 2.0 * z * z - 1.0)))
        eulerangle_xyz[2] = math.atan(-((2.0 * x * y - 2.0 * z * w) / (2.0 * w * w + 2.0 * x * x - 1.0)))
        pass
    elif order == ("yxz"):
        eulerangle_xyz[0] = math.asin(-(2.0 * y * z - 2.0 * x * w))
        eulerangle_xyz[1] = math.atan((2.0 * x * z + 2.0 * y * w) / (2.0 * w * w + 2.0 * z * z - 1.0))
        eulerangle_xyz[2] = math.atan((2.0 * x * y + 2.0 * z * w) / (2.0 * w * w + 2.0 * y * y - 1.0))
        pass
    return eulerangle_xyz

def parse_osc(osc_output):
    osc_output_temp = osc_output
    osc_return = []
#    osc_block_count = 0

    # osc_start_count = 0
    # osc_output_temp = osc_output_temp[osc_start_count::osc_output_temp.__sizeof__() - osc_start_count]

    osc_bundle = osc_output_temp[:8]
    # print(osc_bundle)
    if osc_bundle == b"#bundle\x00":
        osc_timestamp = osc_output_temp[8:16]
        # osc_blocksize = osc_output_temp.__len__() - 16
        osc_blockbytes = osc_output_temp[16:]
        osc_return.extend(parse_osc_bundle(osc_blockbytes))
        pass
    else:
        osc_return.append(osc_output_temp)
        pass

    return osc_return

def parse_osc_bundle(osc_output):
    osc_output_temp = osc_output
    # osc_size = osc_output_temp.__sizeof__()
    # print(osc_size)
    osc_size = osc_output_temp.__len__()
    # print(osc_size)

    osc_return = []
#    osc_block_count = 0

    osc_start_count = 0
    # osc_output_temp = osc_output_temp[osc_start_count::osc_output_temp.__sizeof__() - osc_start_count]

    while osc_start_count < osc_size:
        osc_blocksize = int.from_bytes(osc_output_temp[osc_start_count:osc_start_count + 4], 'big')
        osc_bundle = osc_output_temp[osc_start_count + 4:osc_start_count + 4 + 8]
        if osc_bundle == b"#bundle\x00":
            osc_timestamp = osc_output_temp[osc_start_count + 12:osc_start_count + 12 + 8]
            osc_blockbytes = osc_output_temp[osc_start_count + 20:osc_start_count + 4 + osc_blocksize]
            osc_return.extend(parse_osc_bundle(osc_blockbytes))
            osc_start_count = osc_start_count + 4 + osc_blocksize
            pass
        else:
            osc_blockbytes = osc_output_temp[osc_start_count + 4:osc_start_count + 4 + osc_blocksize]
            osc_return.append(osc_blockbytes)
            osc_start_count = osc_start_count + 4 + osc_blocksize
            pass

    return osc_return
