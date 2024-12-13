import numpy as np
import cv2

# Add these functions after the imports
def find_lane_lines(ll_seg_mask):
    """Find and sort lane lines from segmentation mask"""

    print("find_lane_lines")
    height, width = ll_seg_mask.shape
    
    # Get the bottom half of the image
    bottom_half = ll_seg_mask[height//2:, :]
    
    # Find all lane line points
    lane_points = []
    for col in range(width):
        lane_pixels = np.where(bottom_half[:, col] == 1)[0]
        if len(lane_pixels) > 0:
            # Add points with their x coordinates
            for row in lane_pixels:
                lane_points.append((col, row + height//2))
    
    if not lane_points:
        return None, None
    
    # Cluster points into left and right lanes
    points = np.array(lane_points)
    center_x = width // 2
    
    # Separate left and right points based on image center
    # This assumes camera is mounted at center of vehicle
    left_points = points[points[:, 0] < center_x]
    right_points = points[points[:, 0] >= center_x]
    
    # Fit polynomials if enough points are found
    left_fit = right_fit = None
    if len(left_points) > 10:
        left_fit = np.polyfit(left_points[:, 1], left_points[:, 0], 2)
    if len(right_points) > 10:
        right_fit = np.polyfit(right_points[:, 1], right_points[:, 0], 2)
    
    return left_fit, right_fit


def check_lane_departure(left_fit, right_fit, frame_width, threshold=0.1):
    """Check if vehicle is departing from lane"""
    if left_fit is None or right_fit is None:
        return "UNKNOWN", None
    
    print("check_lane_departure")

    # Calculate lane center at the bottom of the image
    y_eval = frame_width
    left_x = np.polyval(left_fit, y_eval)
    right_x = np.polyval(right_fit, y_eval)
    
    # Calculate center of the lane
    lane_center = (left_x + right_x) / 2
    frame_center = frame_width / 2
    
    # Calculate normalized offset
    offset = (frame_center - lane_center) / frame_width
    
    # Determine departure status
    if abs(offset) < threshold:
        return "CENTER", offset
    elif offset > threshold:
        return "RIGHT_DEPARTURE", offset
    else:
        return "LEFT_DEPARTURE", offset
    


def show_result(dataset, img_det, ll_seg_mask):
    # After processing ll_seg_mask, add:
    if dataset.mode != 'stream':
        # Find lane lines
        left_fit, right_fit = find_lane_lines(ll_seg_mask)
            
        # Check lane departure
        if left_fit is not None and right_fit is not None:
            status, offset = check_lane_departure(left_fit, right_fit, img_det.shape[1])
                
            print(status, offset)
            # Draw lane departure warning
            warning_color = (0, 0, 255) if status.endswith('DEPARTURE') else (0, 255, 0)
            cv2.putText(img_det, f"Status: {status}", (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, warning_color, 2)
            cv2.putText(img_det, f"Offset: {offset:.3f}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, warning_color, 2)
                
            # Draw lane lines
            y_points = np.linspace(img_det.shape[0]//2, img_det.shape[0], 10)
            if left_fit is not None:
                left_x = np.polyval(left_fit, y_points)
                for i in range(len(y_points)-1):
                    pt1 = (int(left_x[i]), int(y_points[i]))
                    pt2 = (int(left_x[i+1]), int(y_points[i+1]))
                    cv2.line(img_det, pt1, pt2, (255, 0, 0), 2)
                
            if right_fit is not None:
                right_x = np.polyval(right_fit, y_points)
                for i in range(len(y_points)-1):
                    pt1 = (int(right_x[i]), int(y_points[i]))
                    pt2 = (int(right_x[i+1]), int(y_points[i+1]))
                    cv2.line(img_det, pt1, pt2, (0, 255, 0), 2) 

    return img_det
