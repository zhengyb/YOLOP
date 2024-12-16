import numpy as np
import torch


def get_the_main_object_by_xpos(det_objs, frame_width, threshold=0.05):

    if len(det_objs) == 0:
        return None

    y2_max = 0
    main_obj = None

    threshold_xmin = frame_width * (0.5 - threshold)
    threshold_xmax = frame_width * (0.5 + threshold)


    for det_obj in det_objs:
        x1, y1, x2, y2, conf, cls = det_obj
        object_center_x = (x1 + x2) / 2

        if object_center_x < threshold_xmin or object_center_x > threshold_xmax:
            continue

        if y2 > y2_max:
            y2_max = y2
            main_obj = det_obj

    return main_obj

def get_the_main_object(left_lane_fit, right_lane_fit, det_objs, frame_width, threshold=0.05):
    """
    A main object is the object that is in the present lane and is the closest to the vehicle(camera).
    This function works only for single-class detection now.
    """

    if len(det_objs) == 0:
        return None

    # Move tensors to CPU if they're on GPU
    if isinstance(det_objs, torch.Tensor):
        det_objs = det_objs.cpu()

    if left_lane_fit is None or right_lane_fit is None:
        return get_the_main_object_by_xpos(det_objs, frame_width, threshold=threshold)      


    y2_max = 0
    main_obj = None

    for det_obj in det_objs:
        x1, y1, x2, y2, conf, cls = det_obj

        print(f"{x1}, {y1}, {x2}, {y2}, {conf}, {cls}")

        object_center_x = (x1 + x2) / 2

        xll = np.polyval(left_lane_fit, y2)
        xlr = np.polyval(right_lane_fit, y2)

        if x2 < xll or x1 > xlr:
            continue

        if x1 < xll and x2 > xlr:
            print(f"Attention! An exception object is detected: {x1}, {y1}, {x2}, {y2}, {conf}, {cls}")
            continue

        # This is a temporary solution to avoid the object that is partially in the lane.
        if object_center_x < xll or object_center_x > xlr:
            print(f"An object that is partially in the lane is detected: {x1}, {y1}, {x2}, {y2}, {conf}, {cls}")
            continue

        if y2 > y2_max:
            y2_max = y2
            main_obj = det_obj

    return main_obj
