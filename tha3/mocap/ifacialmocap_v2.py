import math
import struct

from tha3.mocap.ifacialmocap_constants import BLENDSHAPE_NAMES, HEAD_BONE_X, HEAD_BONE_Y, HEAD_BONE_Z, \
    RIGHT_EYE_BONE_X, RIGHT_EYE_BONE_Y, RIGHT_EYE_BONE_Z, LEFT_EYE_BONE_X, LEFT_EYE_BONE_Y, LEFT_EYE_BONE_Z, \
    HEAD_BONE_QUAT, LEFT_EYE_BONE_QUAT, RIGHT_EYE_BONE_QUAT

from tha3.mocap.ifacialmocap_constants import CHEEK_SQUINT_LEFT, CHEEK_SQUINT_RIGHT

IFACIALMOCAP_PORT = 49983
IFACIALMOCAP_START_STRING = "iFacialMocap_sahuasouryya9218sauhuiayeta91555dy3719|sendDataVersion=v2".encode('utf-8')


def parse_ifacialmocap_v2_pose(ifacialmocap_output):
    output = {}
    parts = ifacialmocap_output.split("|")
    for part in parts:
        part = part.strip()
        if len(part) == 0:
            continue
        if "&" in part:
            components = part.split("&")
            assert len(components) == 2
            key = components[0]
            value = float(components[1]) / 100.0
            if key.endswith("_L"):
                key = key[:-2] + "Left"
            elif key.endswith("_R"):
                key = key[:-2] + "Right"
            if key in BLENDSHAPE_NAMES:
                output[key] = value
        elif part.startswith("=head#"):
            components = part[len("=head#"):].split(",")
            assert len(components) == 6
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
    output[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    return output


def parse_ifacialmocap_v1_pose(ifacialmocap_output):
    output = {}
    parts = ifacialmocap_output.split("|")
    for part in parts:
        part = part.strip()
        if len(part) == 0:
            continue
        if part.startswith("=head#"):
            components = part[len("=head#"):].split(",")
            assert len(components) == 6
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
            assert len(components) == 2
            key = components[0]
            value = float(components[1]) / 100.0
            if key.endswith("_L"):
                key = key[:-2] + "Left"
            elif key.endswith("_R"):
                key = key[:-2] + "Right"
            if key in BLENDSHAPE_NAMES:
                output[key] = value
    output[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    return output

def parse_meowface_pose(ifacialmocap_output):
    output = {}
    
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
            assert len(components) == 2
            key = components[0]
            value = float(components[1]) / 100.0
            if key.endswith("_L"):
                key = key[:-2] + "Left"
            elif key.endswith("_R"):
                key = key[:-2] + "Right"
            if key in BLENDSHAPE_NAMES:
                output[key] = value
        elif part.startswith("=head#"):
            components = part[len("=head#"):].split(",")
            assert len(components) == 6
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
    output[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    output[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]
    return output

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
            # components = part.split("&")
            # assert len(components) == 2
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
