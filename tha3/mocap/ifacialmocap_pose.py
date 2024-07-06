from tha3.mocap.ifacialmocap_constants import BLENDSHAPE_NAMES, HEAD_BONE_X, HEAD_BONE_Y, HEAD_BONE_Z, \
    HEAD_BONE_QUAT, LEFT_EYE_BONE_X, LEFT_EYE_BONE_Y, LEFT_EYE_BONE_Z, LEFT_EYE_BONE_QUAT, RIGHT_EYE_BONE_X, \
    RIGHT_EYE_BONE_Y, RIGHT_EYE_BONE_Z, RIGHT_EYE_BONE_QUAT


def create_default_ifacialmocap_pose():
    data = {}

    for blendshape_name in BLENDSHAPE_NAMES:
        data[blendshape_name] = 0.0

    data["Joy"] = 0.0
    data["Angry"] = 0.0
    data["Sorrow"] = 0.0
    data["Fun"] = 0.0
    data["Surprised"] = 0.0
    data["A"] = 0.0
    data["I"] = 0.0
    data["U"] = 0.0
    data["E"] = 0.0
    data["O"] = 0.0
    data["Blink_L"] = 0.0
    data["Blink_R"] = 0.0
    data["Blink"] = 0.0
    data["LookUp"] = 0.0
    data["LookDown"] = 0.0
    data["LookLeft"] = 0.0
    data["LookRight"] = 0.0
    data["NEUTRAL"] = 0.0

    data[HEAD_BONE_X] = 0.0
    data[HEAD_BONE_Y] = 0.0
    data[HEAD_BONE_Z] = 0.0
    data[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    data[LEFT_EYE_BONE_X] = 0.0
    data[LEFT_EYE_BONE_Y] = 0.0
    data[LEFT_EYE_BONE_Z] = 0.0
    data[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    data[RIGHT_EYE_BONE_X] = 0.0
    data[RIGHT_EYE_BONE_Y] = 0.0
    data[RIGHT_EYE_BONE_Z] = 0.0
    data[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

    return data