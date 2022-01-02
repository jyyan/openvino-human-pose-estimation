#!/usr/bin/env python3
"""
 contributor: edwin2619, Link, Luke

 original sourcecode:
 Copyright (C) 2020-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import sys
import logging

from argparse import ArgumentParser, SUPPRESS
from time import perf_counter
from pathlib import Path

import numpy as np
import cv2
import imutils
from openvino.inference_engine import IENetwork, IECore

sys.path.append(str (Path(__file__).resolve().parents[0]/'common/python'))
import models
import monitors
import math
from images_capture import open_images_capture
from pipelines import get_user_config, AsyncPipeline
from performance_metrics import PerformanceMetrics
from helpers import resolution

#IMG                     = 'img/input.jpg'
# VIDEO                   = 'video/video1.mp4' # use video
VIDEO                   = 4  # open camera
# IR_MODEL                = Path("model/human-pose-estimation-0007.xml")
# IR_MODEL                = Path("model/human-pose-estimation-0006.xml")
IR_MODEL                = Path("model/human-pose-estimation-0002.xml")
RUN_DEVICE              = 'MULTI:CPU,MYRIAD'
# RUN_DEVICE              = 'CPU'
RUN_NUM_STREAMS         = ''
RUN_NUM_THREADS         = None
run_num_infer_requests  = 0
args_OUTPUT_RESOLUTION  = None
OUTPUT_LIMIT            = -1
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of the output file(s) to save.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('-no_show', '--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser

def draw_poses(img, poses, point_score_threshold, output_transform, skeleton=default_skeleton, draw_ellipses=False):
    img = output_transform.resize(img)
    if poses.size == 0:
        return img
    stick_width = 4

    log.info('Poses size: {}'.format(poses.size))

    img_limbs = np.copy(img)
    for pose in poses:

        points = pose[:, :2].astype(np.int32)
        points = output_transform.scale(points)
        points_scores = pose[:, 2]
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                if draw_ellipses:
                    middle = (points[i] + points[j]) // 2
                    vec = points[i] - points[j]
                    length = np.sqrt((vec * vec).sum())
                    angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                    polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)),
                                                angle, 0, 360, 1)
                    cv2.fillConvexPoly(img_limbs, polygon, colors[j])
                else:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=stick_width)


        #Draw body axis
        cv2.line(img_limbs, tuple((points[12]+points[11])//2), tuple((points[5]+points[6])//2), color=colors[j], thickness=stick_width)
        #Calculate body angle
        vec = ((points[12]+points[11])//2) - ((points[5]+points[6])//2)
        angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi )
        #Draw body angle
        cv2.putText(img,str(angle),(10,120),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)
        #Show message if body angle not correct
        if (abs(90 - angle) >= 10):
            cv2.putText(img,'humpback',(10,160),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)

    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img

def print_raw_results(poses, scores):
    log.info('Poses:')
    for pose, pose_score in zip(poses, scores):
        pose_str = ' '.join('({:.2f}, {:.2f}, {:.2f})'.format(p[0], p[1], p[2]) for p in pose)
        log.info('{} | {:.2f}'.format(pose_str, pose_score))

def main():
    args = build_argparser().parse_args()
    metrics = PerformanceMetrics()
    log.info('Initialize Inference Engine....')
    ie =IECore()

    plugin_config = get_user_config(RUN_DEVICE, RUN_NUM_STREAMS, RUN_NUM_THREADS)

    start_time = perf_counter()
    #cap = open_images_capture(IMG, False)
    #frame = cap.read()
    cap = cv2.VideoCapture(VIDEO)
    ret, frame = cap.read()
    #frame = cv2.imread(IMG)

    if frame is None :
        raise RuntimeError('Can not read an image from the input')
    aspect_ratio = frame.shape[1]/frame.shape[0]
    print(frame.shape)
    log.info('Loading Network....')
    model = models.HpeAssociativeEmbedding(ie, IR_MODEL, target_size=None, aspect_ratio=aspect_ratio,prob_threshold=0.1 )
    hpe_pipeline = AsyncPipeline(ie, model, plugin_config, device=RUN_DEVICE, max_num_requests=run_num_infer_requests)

    log.info('Starting inference...')
    # frame = imutils.resize(frame, width=300)
    hpe_pipeline.submit_data(frame, 0, {'frame': frame, 'start_time' : start_time})
    next_frame_id =1
    next_frame_id_to_show =0

    output_transform = models.OutputTransform(frame.shape[:2],args_OUTPUT_RESOLUTION )

    output_resolution = (frame.shape[1], frame.shape[0])

    presenter = monitors.Presenter('',55, (round(output_resolution[0]/4), round(output_resolution[1]/8 )))
    video_writer= cv2.VideoWriter()
    # if not video_writer.open('output.mp4', cv2.VideoWriter_fourcc(*'mp4v') ,30.0,output_resolution):
    # if not video_writer.open('', cv2.VideoWriter_fourcc(*'MJPG') , cap.fps(),output_resolution):
        # raise RuntimeError("Can't open video writer")

    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'), cap.fps(),
            output_resolution):
        raise RuntimeError("Can't open video writer")


    while True:
        if hpe_pipeline.callback_exceptions:
            raise hep_pipeline.callback_exceptions[0]
        # Process all completed requests
        results = hpe_pipeline.get_result(next_frame_id_to_show)
        if results:
            # print("next_frame_id_to_show", next_frame_id_to_show)
            (poses, scores), frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if len(poses) and args.raw_output_message:
               print_raw_results(poses, scores)

            presenter.drawGraphs(frame)
            frame = draw_poses(frame, poses, 0.1, output_transform)
            metrics.update(start_time, frame)
            if video_writer.isOpened() and (OUTPUT_LIMIT <=0 or next_frame_id_to_show <= OUTPUT_LIMIT-1):
                video_writer.write(frame)
            next_frame_id_to_show +=1


            if not args.no_show:
                cv2.imshow('Pose estimation results', frame)
                key = cv2.waitKey(1)

                ESC_KEY = 27
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
                presenter.handleKey(key)
            continue

        if hpe_pipeline.is_ready():
            start_time = perf_counter()
            ret,frame = cap.read()
            if frame is None:
                break
            hpe_pipeline.submit_data(frame, next_frame_id, {'frame':frame, 'start_time':start_time})
            next_frame_id +=1
        else:
            hpe_pipeline.await_any()
    hpe_pipeline.await_all()
    #print("Debug ")
    # Process completed requests
    for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
        results = hpe_pipeline.get_result(next_frame_id_to_show)
        # print(next_frame_id_to_show)
        while results is None:
            results = hpe_pipeline.get_result(next_frame_id_to_show)
        #print(results)
        (poses, scores), frame_meta = results
        frame = frame_meta['frame']
        start_time = frame_meta['start_time']

        # if len(poses) and args.raw_output_message:
           # print_raw_results(poses, scores)

        presenter.drawGraphs(frame)
        frame = draw_poses(frame, poses, 0.1, output_transform)
        metrics.update(start_time, frame)
        if video_writer.isOpened() and (OUTPUT_LIMIT <= 0 or next_frame_id_to_show <= OUTPUT_LIMIT-1):
            video_writer.write(frame)
    metrics.print_total()
    print(presenter.reportMeans())

if __name__ == '__main__':
    sys.exit(main() or 0)
