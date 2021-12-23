import sys
import logging

from argparse import ArgumentParser, SUPPRESS
from time import perf_counter
from pathlib import Path

import numpy as np
import cv2
from openvino.inference_engine import IENetwork, IECore

sys.path.append(str (Path(__file__).resolve().parents[0]/'common/python'))
import models
import monitors
from images_capture import open_images_capture
from pipelines import get_user_config, AsyncPipeline
from performance_metrics import PerformanceMetrics
from helpers import resolution

IMG                     = 'img/input.jpg'
IR_MODEL                = Path("model/human-pose-estimation-0007.xml")
RUN_DEVICE              = 'CPU'
RUN_NUM_STREAMS         = ''
RUN_NUM_THREADS         = None
run_num_infer_requests  = 0
args_OUTPUT_RESOLUTION  = None

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

def draw_poses(img, poses, point_score_threshold, output_transform, skeleton=default_skeleton, draw_ellipses=False):
    img = output_transform.resize(img)
    if poses.size == 0:
        return img
    stick_width = 4

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
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img



def main():
    
    log.info('Initialize Inference Engine....')
    ie =IECore()

    plugin_config = get_user_config(RUN_DEVICE, RUN_NUM_STREAMS, RUN_NUM_THREADS)
    
    start_time = perf_counter()
    #cap = open_images_capture(IMG, False)
    #frame = cap.read()
    frame = cv2.imread(IMG)
    if frame is None :
        raise RuntimeError('Can not read an image from the input')
    aspect_ratio = frame.shape[1]/frame.shape[0]
    log.info('Loading Network....')
    model = models.HpeAssociativeEmbedding(ie, IR_MODEL, target_size=None, aspect_ratio=aspect_ratio,prob_threshold=0.1 )
    hpe_pipeline = AsyncPipeline(ie, model, plugin_config, device=RUN_DEVICE, max_num_requests=run_num_infer_requests)
    
    log.info('Starting inference...')
    hpe_pipeline.submit_data(frame, 0, {'frame': frame, 'start_time' : start_time})
    next_frame_id =1
    next_frame_id_to_show =0

    output_transform = models.OutputTransform(frame.shape[:2],args_OUTPUT_RESOLUTION )

    output_resolution = (frame.shape[1], frame.shape[0])

    presenter = monitors.Presenter('',55, (round(output_resolution[0]/4), round(output_resolution[1]/8 )))
    video_writer= cv2.VideoWriter()
    if not video_writer.open('output.jpg', cv2.VideoWriter_fourcc(*'MJPG') ,20.0,output_resolution):
        raise RuntimeError("Can't open video writer")

    while True:
        if hpe_pipeline.callback_exceptions:
            raise hep_pipeline.callback_exceptions[0]
        # Process all completed requests
        results = hpe_pipeline.get_result(next_frame_id_to_show)
        if results:
            print("Debug loop")
            (poses, scroes), frmae_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            presenter.drawGraphs(frame)
            frame = draw_poses(frame, poses, 0.1,output_transform) 

            if video_writer.isOpened() and (1000 <=0 or next_frame_id_to_show <= 999):
                video_writer.write(frame)
            next_frame_id_to_show +=1

            continue

        if hpe_pipeline.is_ready():
            break
        else:
            hpe_pipeline.await_any()
    hpe_pipeline.await_all()
    print("Debug ")
    # Process completed requests
    for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
        results = hpe_pipeline.get_result(next_frame_id_to_show)
        print(next_frame_id_to_show)
        while results is None:
            results = hpe_pipeline.get_result(next_frame_id_to_show)
        print(results)
        (poses, scores), frame_meta = results
        frame = frame_meta['frame']
        start_time = frame_meta['start_time']

        #if len(poses) and args.raw_output_message:
        #    print_raw_results(poses, scores)

        presenter.drawGraphs(frame)
        frame = draw_poses(frame, poses, 0.1, output_transform)
        #metrics.update(start_time, frame)
        if video_writer.isOpened() and (1000 <= 0 or next_frame_id_to_show <= 999):
            video_writer.write(frame)
        

if __name__ == '__main__':
    sys.exit(main() or 0) 
