import numpy as np
import cv2
from sklearn.cluster import KMeans
import math

def cartesian_to_polar(x1, y1, x2, y2):
    """
    将笛卡尔坐标系下的直线转换为极坐标系中的 (rho, theta)。

    参数：
        x1, y1: 直线上的第一个点
        x2, y2: 直线上的第二个点

    返回：
        rho: 直线到原点的垂直距离
        theta: 垂直于直线的向量与x轴正方向的夹角（弧度制，范围 [0, pi)）
    """
    # 计算直线的法向量
    dx = x2 - x1
    dy = y2 - y1

    # 法向量的垂直方向
    nx = -dy
    ny = dx

    # 法向量与原点到直线垂线交点的方向一致
    rho = abs(nx * x1 + ny * y1) / math.sqrt(nx**2 + ny**2)

    # 计算 theta，注意 atan2 的返回值范围是 [-pi, pi]
    theta = math.atan2(ny, nx)

    # 将 theta 映射到 [0, pi) 区间
    if theta < 0:
        theta += math.pi

    return rho, theta

def merge_lines(polar_lines, rho_threshold=10, theta_threshold=0.1):
    merged_lines = []
    for line in polar_lines:
        print(line)
        rho = line[0]
        theta = line[1]
        found = False
        for merged_line in merged_lines:
            mrho, mtheta, mcount = merged_line
            if abs(rho - mrho) < rho_threshold and abs(theta - mtheta) < theta_threshold:
                merged_line[0] = (mrho * mcount + rho) / (mcount + 1)
                merged_line[1] = (mtheta * mcount + theta) / (mcount + 1)
                merged_line[2] += 1
                found = True
                break
        if not found:
            merged_lines.append([rho, theta, 1])
    return merged_lines

def polar_to_cartesian(rho, theta, length=1000):
    x1 = int(rho * math.cos(theta) - length * math.sin(theta))
    y1 = int(rho * math.sin(theta) + length * math.cos(theta))
    x2 = int(rho * math.cos(theta) + length * math.sin(theta))
    y2 = int(rho * math.sin(theta) - length * math.cos(theta))
    return x1, y1, x2, y2      


def polar_to_slope_intercept(rho, theta):
    slope = 0
    intercept = 0

    x1, y1, x2, y2 = polar_to_cartesian(rho, theta)
    if x2 == x1:  # Skip vertical lines
        return None, None, x1, y1, x2, y2
            
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    return slope, intercept, x1, y1, x2, y2




def find_lane_lines_multi(ll_seg_mask, rho=1, theta=np.pi/180, threshold=20, 
                         min_line_length=100, max_line_gap=50, img_det=None):
    """Find and sort multiple lane lines using Hough transform
    
    Args:
        ll_seg_mask: Binary segmentation mask
        rho: Distance resolution of Hough accumulator
        theta: Angle resolution of Hough accumulator
        threshold: Minimum number of votes needed to consider a line
        min_line_length: Minimum length of line
        max_line_gap: Maximum gap between line segments
    """
    height, width = ll_seg_mask.shape
    center_x = width // 2
    
    # Convert mask to uint8 for cv2.HoughLinesP
    binary = ll_seg_mask.astype(np.uint8) * 255
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(binary, rho, theta, threshold, 
                           minLineLength=min_line_length,
                           maxLineGap=max_line_gap)
    if lines is None:
        return None, None, [], []
    print(f"lines1: {len(lines)}")
    #print(lines)


    line_image = np.copy(ll_seg_mask) * 0

    print(lines[0])

    #polar_lines = [cartesian_to_polar(x1, y1, x2, y2) for x1, y1, x2, y2 in [line for line in lines]]
    polar_lines = []
    for line in lines:
        print(f"line: {line}")
        line_image = cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 0, 0), 2)
        rho, theta = cartesian_to_polar(line[0][0], line[0][1], line[0][2], line[0][3])
        polar_lines.append([rho, theta])
    merged_polar_lines = merge_lines(polar_lines)

    lines = []
    print(f"Merged lines: {len(merged_polar_lines)}")
    #for mline in merged_polar_lines:
    for i in range(len(merged_polar_lines)):
        mline = merged_polar_lines[i]
        print(f"{mline}")
        rho = mline[0]
        theta = mline[1]
        x1, y1, x2, y2 = polar_to_cartesian(rho, theta)
        slope, intercept, x1, y1, x2, y2 = polar_to_slope_intercept(rho, theta)
        lines.append([x1, y1, x2, y2])
        # rho, theta, count, slope, intercept, x1, y1, x2, y2
        merged_polar_lines[i] = mline + [slope, intercept, x1, y1, x2, y2]
        print(f"{merged_polar_lines[i]}")
        #line_image = cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # draw lines and ll_seg_mask on a blank image

    #cv2.imwrite("./debug_hough.jpg", line_image)
 


    # Separate left and right lines based on slope
    left_lines = []
    most_left_line_x = 0 - width
    most_left_line = None
    right_lines = []
    most_right_line_x = width * 2
    most_right_line = None

    
    for line in merged_polar_lines:
        rho, theta, count, slope, intercept, x1, y1, x2, y2 = line
        if x2 == x1:  # Skip vertical lines
            continue
        bottom_x = (height - intercept) / slope
        print(f"bottom x: {bottom_x}")
        # Filter based on slope and position
        if 0.3 < abs(slope) < 2.0:  # Reasonable slope range for lane lines
            if bottom_x < center_x and slope < 0:
                left_lines.append(line)
                if bottom_x > most_left_line_x:
                    most_left_line = line
                    most_left_line_x = bottom_x
            elif bottom_x > center_x and slope > 0:
                right_lines.append(line)
                if bottom_x < most_right_line_x:
                    most_right_line = line
                    most_right_line_x = bottom_x
    
    print(f"Most left line ({most_left_line}), bottom x {most_left_line_x}")
    print(f"Most right line ({most_right_line}), bottom x {most_right_line_x}")

    # draw the left lane with blue and the right lane with green color
    if most_left_line is not None:
        _, _, x1, y1, x2, y2 = most_left_line[-6:]
        cv2.line(img_det, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue color for left lane

    if most_right_line is not None:
        _, _, x1, y1, x2, y2 = most_right_line[-6:]
        cv2.line(img_det, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green color for right lane

    #cv2.imwrite("./debug_hough.jpg", line_image)
        # Find most relevant lanes
    left_fit = most_left_line
    right_fit = most_right_line
    
    return left_fit, right_fit, line_image

def cluster_and_fit_lanes(lines, height, is_left=True, n_clusters=2):
    """Cluster lines into lanes and fit polynomials"""
    if not lines:
        return []
    
    print(f"lines: {len(lines)}")
        
    # Extract slopes and intercepts for clustering
    slopes_intercepts = np.array([(slope, intercept) for slope, intercept, *_ in lines])
    
    if len(slopes_intercepts) < n_clusters:
        n_clusters = len(slopes_intercepts)
    
    # Cluster lines using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(slopes_intercepts)
    
    lanes = []
    for cluster_id in range(n_clusters):
        cluster_lines = [lines[i] for i in range(len(lines)) if clusters[i] == cluster_id]
        if not cluster_lines:
            continue
            
        # Generate points from lines for polynomial fitting
        points = []
        for _, _, x1, y1, x2, y2 in cluster_lines:
            points.extend([(x1, y1), (x2, y2)])
            
        points = np.array(points)
        if len(points) < 2:
            continue
            
        try:
            # Fit quadratic polynomial
            lane_fit = np.polyfit(points[:, 1], points[:, 0], 2)
            lanes.append({
                'fit': lane_fit,
                'points': points,
                'center_y': np.mean(points[:, 1])
            })
        except np.linalg.LinAlgError:
            continue
            
    print(f"lanes: {len(lanes)}")
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

def check_lane_departure_multi(left_fit, right_fit, frame_width, threshold=0.1):
    """Check if vehicle is departing from lane"""
    if left_fit is None or right_fit is None:
        return "UNKNOWN", None, None
    
    # Calculate lane positions at the bottom of the image
    y_eval = frame_width
    left_x = np.polyval(left_fit, y_eval)
    right_x = np.polyval(right_fit, y_eval)
    
    # Calculate lane width and center
    lane_width = right_x - left_x
    lane_center = (left_x + right_x) / 2
    frame_center = frame_width / 2
    
    # Calculate normalized offset
    offset = (frame_center - lane_center) / frame_width
    
    # Calculate lane width ratio
    expected_lane_width = frame_width * 0.4
    width_ratio = lane_width / expected_lane_width
    
    # Determine departure status
    if width_ratio < 0.7 or width_ratio > 1.3:
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
    rho = 3
    theta = 2 * np.pi/180
    threshold = 650
    min_line_length = 100
    max_line_gap = 300

    left_fit, right_fit, line_image = find_lane_lines_multi(ll_seg_mask,
                        rho=rho,
                        theta=theta,
                        threshold=threshold,
                        min_line_length=min_line_length,
                        max_line_gap=max_line_gap, img_det=img_det)
#    img_det = visualize_lanes(img_det, None, None, left_fit, right_fit)
    return img_det