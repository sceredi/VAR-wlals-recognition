import cv2
from app.utilities.utils import video_to_img


def preprocess(video_src):
    video_capture = cv2.VideoCapture(video_src)
    tracking_rgb_frames = []
    tracking_back_projections = []
    ret, frame = video_capture.read()
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    roi_rect = [0, 0, frame.shape[0] - 1, frame.shape[1] - 1]

    while True:
        # 1
        ret, frame = video_capture.read()

        if ret:
            # 2
            # if resize_factor!=1:
            #   frame=cv2.resize(frame,(resized_width,resized_height),cv2.INTER_LINEAR)

            # 3
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 4
            h_range = [0, 20]
            s_range = [90, 180]
            v_range = [100, 165]

            bin_count = 100
            mask = cv2.inRange(hsv_frame, (h_range[0], s_range[0], v_range[0]), (h_range[1], s_range[1], v_range[1]))
            roi_h_hist = cv2.calcHist([hsv_frame], [0], mask, [bin_count], h_range)
            back_proj = cv2.calcBackProject([hsv_frame], [0], roi_h_hist, h_range, 1)

            # 5
            frame_mask = cv2.inRange(hsv_frame, (h_range[0], s_range[0], v_range[0]),
                                     (h_range[1], s_range[1], v_range[1]))

            # 6
            back_proj = cv2.bitwise_and(back_proj, back_proj, mask=frame_mask)

            # 7
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            _, roi_rect = cv2.meanShift(back_proj, roi_rect, term_crit)

            frame_with_roi = cv2.rectangle(frame, (roi_rect[0], roi_rect[1]),
                                           (roi_rect[0] + roi_rect[2], roi_rect[1] + roi_rect[3]), (0, 0, 255), 1)
            tracking_rgb_frames.append(cv2.cvtColor(frame_with_roi, cv2.COLOR_BGR2RGB))

            back_proj_with_roi = cv2.rectangle(cv2.cvtColor(back_proj, cv2.COLOR_GRAY2RGB), (roi_rect[0], roi_rect[1]),
                                               (roi_rect[0] + roi_rect[2], roi_rect[1] + roi_rect[3]), (255, 0, 0), 1)
            tracking_back_projections.append(back_proj_with_roi)
        else:
            break
    fg = video_to_img(tracking_back_projections)

    print('Numero di frame elaborati:', len(tracking_rgb_frames))
    return fg
