import numpy as np
from sklearn.cluster import DBSCAN
import cv2

def find_lane_lines_multi(ll_seg_mask, min_points=10, eps=20, min_samples=5):
    """Find and sort multiple lane lines from segmentation mask"""
    height, width = ll_seg_mask.shape
    
    # Get the bottom half of the image
    bottom_half = ll_seg_mask[height//2:, :]
    
    # Find all lane line points
    lane_points = []
    for col in range(width):
        lane_pixels = np.where(bottom_half[:, col] == 1)[0]
        if len(lane_pixels) > 0:
            for row in lane_pixels:
                lane_points.append((col, row + height//2))
    
    if not lane_points:
        return None, None, None, None
    
    points = np.array(lane_points)
    center_x = width // 2
    
    # Separate left and right points based on image center
    # This doesn't work for b1c81faa-3df17267.jpg
    left_points = points[points[:, 0] < center_x]
    right_points = points[points[:, 0] >= center_x]
    
    # Find multiple lanes using DBSCAN clustering
    left_lanes = cluster_lane_points(left_points, eps=eps, min_samples=min_samples)
    print(f"Left lanes after cluster: {len(left_lanes)}")
    display_poly_lanes(left_lanes)
    right_lanes = cluster_lane_points(right_points, eps=eps, min_samples=min_samples)
    print(f"Right lanes after cluster: {len(right_lanes)}")
    display_poly_lanes(right_lanes)

    # Find the most relevant lanes (closest to vehicle)
    left_fit = find_closest_lane(left_lanes, center_x, is_left=True)
    print(f"Left fit: {left_fit}")
    right_fit = find_closest_lane(right_lanes, center_x, is_left=False)
    print(f"Right fit: {right_fit}")
    
    return left_fit, right_fit, left_lanes, right_lanes

def display_poly_lanes(poly_lanes):
    i = 0
    for lane in poly_lanes:
        i = i+1
        print(f"Lane {i}: {lane['fit']}; point count {len(lane['points'])}; {lane['center_y']}")


def cluster_lane_points(points, eps=20, min_samples=5, coeffs0_th=0.01, coeffs1_th=0.1):
    """Cluster lane points into multiple lanes using DBSCAN"""
    if len(points) < min_samples:
        return []
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    # Separate clusters and fit polynomials
    lanes = []
    for label in set(labels):
        if label == -1:  # Skip noise points
            continue
        
        # Get points for this lane
        lane_points = points[labels == label]
        if len(lane_points) > 10:
            # Fit polynomial to lane points
            lane_fit = np.polyfit(lane_points[:, 1], lane_points[:, 0], 2)
            if abs(lane_fit[0]) > coeffs0_th or abs(lane_fit[1]) < coeffs1_th:
                print(f"filter this line: {lane_fit}")
                continue
            lanes.append({
                'fit': lane_fit,
                'points': lane_points,
                'center_y': np.mean(lane_points[:, 1])  # Use for sorting
            })
    
    return lanes

def find_closest_lane(lanes, center_x, is_left=True):
    """Find the most relevant lane (closest to vehicle)"""
    if not lanes:
        return None
    
    # Sort lanes by their x-position at the bottom of the image
    def get_lane_position(lane):
        y_eval = lane['center_y']
        x_pos = np.polyval(lane['fit'], y_eval)
        return abs(x_pos - center_x)
    
    sorted_lanes = sorted(lanes, key=get_lane_position)
    return sorted_lanes[0]['fit']

def check_lane_departure_multi(left_fit, right_fit, frame_width, frame_height, threshold=0.5, width_ratio_min=0.7, width_ratio_max=2):
    """Check if vehicle is departing from lane"""
    if left_fit is None or right_fit is None:
        return "UNKNOWN", None, None
    
    # Calculate lane positions at the bottom of the image
    y_eval = frame_height
    left_x = np.polyval(left_fit, y_eval)
    right_x = np.polyval(right_fit, y_eval)
    
    # Calculate lane width and center
    lane_width = right_x - left_x
    lane_center = (left_x + right_x) / 2
    frame_center = frame_width / 2
    
    # Calculate normalized offset
    offset = (frame_center - lane_center) / lane_width
    
    # Calculate lane width ratio (can be used to validate detection)
    expected_lane_width = frame_width * 0.4  # Typical lane width
    width_ratio = abs(lane_width / expected_lane_width)
    
    print(f"width: {lane_width}/{expected_lane_width}; offset: {lane_center} -> {frame_center}")

    # Determine departure status with width validation
    if width_ratio_min < 0.7 or width_ratio_max > 2:
        return "INVALID_LANE_WIDTH", offset, width_ratio
    elif abs(offset) < threshold:
        return "CENTER", offset, width_ratio
    elif offset > threshold:
        return "RIGHT_DEPARTURE", offset, width_ratio
    else:
        return "LEFT_DEPARTURE", offset, width_ratio

def visualize_lanes(image, all_left_lanes, all_right_lanes, active_left, active_right):
    """Visualize all detected lanes and highlight the active ones"""
    vis_img = image.copy()
    
    # Draw all detected lanes in light gray
    y_points = np.linspace(image.shape[0]//2, image.shape[0], 10)
    
    # Draw inactive lanes
    all_lanes = []
    if all_left_lanes is not None:
        all_lanes += [(all_left_lanes, True)]
    if all_right_lanes is not None:
        all_lanes += [(all_right_lanes, False)]

    for lanes, is_left in all_lanes:
        for lane in lanes:
            fit = lane['fit']
            x_points = np.polyval(fit, y_points)
            points = np.column_stack((x_points, y_points)).astype(np.int32)
            color = (200, 200, 200) if is_left else (100, 100, 100)
            cv2.polylines(vis_img, [points], False, color, 2)

    # Draw active lanes in bold colors
    if active_left is not None:
        x_points = np.polyval(active_left, y_points)
        points = np.column_stack((x_points, y_points)).astype(np.int32)
        cv2.polylines(vis_img, [points], False, (255, 0, 0), 4)
    
    if active_right is not None:
        x_points = np.polyval(active_right, y_points)
        points = np.column_stack((x_points, y_points)).astype(np.int32)
        cv2.polylines(vis_img, [points], False, (0, 255, 0), 4)
    
    return vis_img


def show_result(dataset, img_det, ll_seg_mask):
    # After processing ll_seg_mask, add:
    if dataset.mode != 'stream':
        # Find lane lines
        left_fit, right_fit, all_left_lanes, all_right_lanes = find_lane_lines_multi(ll_seg_mask)
            
        # draw all lanes in the img_det
        img_det = visualize_lanes(img_det, None, all_right_lanes, None, None)

        # Check lane departure
        if left_fit is not None and right_fit is not None:
            status, offset, width_ratio = check_lane_departure_multi(left_fit, right_fit, img_det.shape[1], img_det.shape[0])
                
            print(status, offset, width_ratio)
            # Draw lane departure warning
            warning_color = (0, 0, 255) if status.endswith('DEPARTURE') else (0, 255, 0)
            cv2.putText(img_det, f"Status: {status}", (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, warning_color, 2)
            cv2.putText(img_det, f"Offset: {offset:.3f}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, warning_color, 2)
                
            if False:    
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