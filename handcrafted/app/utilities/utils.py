import os
import uuid

import cv2
import numpy as np


def create_mp4_video_from_frames(frames, fps, video_path):
    temp_video_path = "tempfile.mp4"
    compressed_path = "{}.mp4".format(str(uuid.uuid4()))

    size = (frames[0].shape[1], frames[0].shape[0])
    out = cv2.VideoWriter(
        temp_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size  # type: ignore
    )

    for i in range(len(frames)):
        out.write(frames[i][..., ::-1].copy())
    out.release()

    os.system(f"ffmpeg -i {temp_video_path} -vcodec libx264 {compressed_path}")

    os.remove(temp_video_path)

    os.replace(compressed_path, video_path)

    return compressed_path


def draw_connected_components(labels):
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])  # type: ignore

    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    labeled_img[label_hue == 0] = 0

    return labeled_img


def draw_detected_objects(
    image,
    detected_bbs,
    detected_centroids,
    contours=None,
    detected_colors=(0, 0, 255),
):
    if contours is not None:
        image_with_detected_objects = cv2.polylines(
            image.copy(), contours, True, (0, 255, 0), 1
        )
    else:
        image_with_detected_objects = image.copy()

    for i in range(len(detected_bbs)):
        if type(detected_colors) is list:
            color = detected_colors[i]
        else:
            color = detected_colors

        image_with_detected_objects = cv2.rectangle(
            image_with_detected_objects,
            detected_bbs[i][0],
            detected_bbs[i][1],
            color,
            1,
        )
        image_with_detected_objects = cv2.circle(
            image_with_detected_objects, detected_centroids[i], 3, color, -1
        )

    return image_with_detected_objects


def draw_tracked_objects(
    image, tracked_bbs, tracks, contours=None, tracked_colors=(0, 0, 255)
):
    if contours is not None:
        image_with_tracked_objects = cv2.polylines(
            image.copy(), contours, True, (0, 255, 0), 1
        )
    else:
        image_with_tracked_objects = image.copy()

    for i in range(len(tracked_bbs)):
        if type(tracked_colors) is list:
            color = tracked_colors[i]
        else:
            color = tracked_colors

        image_with_tracked_objects = cv2.rectangle(
            image_with_tracked_objects,
            tracked_bbs[i][0],
            tracked_bbs[i][1],
            color,
            1,
        )
        for j in range(len(tracks[i]) - 1):
            image_with_tracked_objects = cv2.line(
                image_with_tracked_objects,
                tracks[i][j],
                tracks[i][j + 1],
                color,
                1,
            )

    return image_with_tracked_objects


def compute_iou(bb1, bb2):
    bb1_x1 = bb1[0][0]
    bb1_y1 = bb1[0][1]
    bb1_x2 = bb1[1][0]
    bb1_y2 = bb1[1][1]

    bb2_x1 = bb2[0][0]
    bb2_y1 = bb2[0][1]
    bb2_x2 = bb2[1][0]
    bb2_y2 = bb2[1][1]

    x_left = max(bb1_x1, bb2_x1)
    y_top = max(bb1_y1, bb2_y1)
    x_right = min(bb1_x2, bb2_x2)
    y_bottom = min(bb1_y2, bb2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1_x2 - bb1_x1) * (bb1_y2 - bb1_y1)
    bb2_area = (bb2_x2 - bb2_x1) * (bb2_y2 - bb2_y1)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def video_to_img(back_projs):
    estimated_fg = []
    estimated_fg.append(back_projs[0])
    fg = back_projs[0]
    k = 1
    while k < len(back_projs):
        estimated_fg.append(
            cv2.addWeighted(
                estimated_fg[k - 1],
                1 - (1 / len(back_projs)),
                back_projs[k],
                1 / len(back_projs),
                0,
            )
        )
        fg = np.add(fg, estimated_fg[k])
        k += 1
    return fg
