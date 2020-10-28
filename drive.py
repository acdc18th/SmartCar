import numpy as np #
import cv2 #

def detect_canny(frame):
    #
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV",hsv)

    #
    lower_black=(0,0,0)
    upper_black=(255,255,80)
    mask=cv2.inRange(hsv,lower_black,upper_black)
    mask_result=cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow("Mask Result",mask_result)

    #
    canny=cv2.Canny(mask,200,400)
    canny_result=canny[50:200,0:320]

    return canny_result

def detect_line_segments(canny_result):
    angle=np.pi/180
    line_segments=cv2.HoughLinesP(canny_result,1,angle,10,
                                  np.array([]),minLineLength=8,maxLineGap=4)
    return line_segments

def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        #logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                #logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    #logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def detect_lane(frame):
    canny_result=detect_canny(frame)
    line_segments=detect_line_segments(canny_result)
    lane_lines=average_slope_intercept(frame, line_segments)

    return lane_lines

cam=cv2.VideoCapture(0) #

while cam.isOpened():
    success,frame=cam.read() #

    #
    if not success:
        continue
    cv2.imshow('Original',frame)

    lane_lines=detect_lane(frame)
    line_image=np.zeros_like(frame)

    if lane_lines is not None:
        for line in lane_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),1,1)
    line_image=cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow("Lane Lines",line_image)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cam.release()
cv2.destroyAllWindows()
