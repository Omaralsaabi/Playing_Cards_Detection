# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import torch

# TODO: added
from collections import deque
import math
from collections import Counter
import keyboard
from Tarneeb.t_game import play, start

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from math import ceil


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=True,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # TODO: Modifications 3
    n_timer = 0
    cat_timer = {}
    actual_inputs = []
    card_order = []
    # pts = deque(maxlen=32)
    final_coords = {}
    card_coords = {}


    tarneeb_input = {}
    tarneeb_rounds = 0
    tarneeb_data = {}
    tarneeb_p_names = None
    tarneeb_game_num = 1
    tarneeb_end = False
    started = False
    tarneeb_data['teamFinalPoints'] = {'team 1': 0, 'team 2': 0}


    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, font_size=12, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # TODO: MODIFIED
                input_names = {}

                # Print results
                n_total = 0

                # annotate_result = 'Cards\n' # LATER
                annotate_result = ''


                for c in det[:, 5].unique():
                    # n: number of cards of some category
                    # names[int(c)]: the category
                    if 'ZZCard' in names[int(c)]:
                        n = (det[:, 5] == c).sum()
                        input_names[int(c)] = 'BackCard'
                    elif 'Zj' in names[int(c)]:
                        n = ceil((det[:, 5] == c).sum() / 2)
                        input_names[int(c)] = 'Joker'
                    else:
                        input_names[int(c)] = names[int(c)]
                        n = ceil((det[:, 5] == c).sum() / 2)  # detections per class
                        # n = (det[:, 5] == c).sum()
                    key = f'{input_names[int(c)]} {n}'
                    if key in cat_timer.keys():
                        cat_timer[key]  += 1
                    else:
                        cat_timer[key] = 0
                        cat_timer[key] += 1

                    n_total += n
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)} {det[:, 5].unique().sum()}, "  # add to string
                    s += f"{n} {names[int(c)]}, "  # add to string
                    # annotate_result += f'{names[int(c)]} -> {n}\n' # LATER
                # n_timer += 1

                # if n_timer == 200:
                #     n_timer = 0
                #     actual_inputs = []
                #     for cat_i in cat_timer.keys():
                #         if cat_timer[cat_i] >= 70:
                #             actual_inputs.append(cat_i)
                #     cat_timer = {}
                #     s += f"total cards: {n_total} "
                # else:
                #     s += f"total cards: {n_total} "
                annotate_result += f"total cards: {n_total} "

                # Write results
                coords = {}
                last_cate = ""
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # TODO: MODIFIED 2
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (input_names[c] if True else f'{input_names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # print(xyxy)
                        xyxy_temp = list(map(lambda tensor_val: tensor_val.item(), xyxy))
                        # print(xyxy_temp)
                        new_x, new_y = (xyxy_temp[0] + xyxy_temp[2])/2, (xyxy_temp[1] + xyxy_temp[3])/2
                        if input_names[c] not in coords:
                            coords[input_names[c]] = []
                        coords[input_names[c]].append((new_x, new_y))

                        # LOGGER.info(f"{input_names[c]}  ({new_x},{new_y})")
                        # if last_cate == input_names[c]:
                        #     final_x, final_y = (new_x + final_x)/2, (new_y + final_y)/2
                        # else:
                        #     final_x, final_y = new_x, new_y
                        # last_cate = input_names[c]

                # LOGGER.info(f"FINAL {input_names[c]}  ({final_x},{final_y})")


                        # radius = math.sqrt((new_x-xyxy_temp[0])**2 + (new_y-xyxy_temp[1])**2) * 2
                        # print('new_x: {}, new_y {}, radius {}'.format(new_x, new_y, radius))
                        #
                        # cv2.circle(frame, (int(new_x), int(new_y)), int(radius),
                        #            (0, 255, 255), 2)
                        # cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        # pts.appendleft(center)
                        # print(xyxy)
                        # print('box coordinates: ', end='')
                        # for xyxy_i in xyxy:
                        #     print(f'{xyxy_i.item()}', end=' ')
                        # print(type(xyxy[0]))
                        # only proceed if at least one contour was found

                        # TODO: modifications failed
                        # cnts = xyxy
                        # cnts = cv2.findContours(xyxy, cv2.RETR_EXTERNAL,
                        #                         cv2.CHAIN_APPROX_SIMPLE)
                        # cnts = imutils.grab_contours(cnts)
                        # center = None
                        # if len(cnts) > 0:
                        #     # find the largest contour in the mask, then use
                        #     # it to compute the minimum enclosing circle and
                        #     # centroid
                        #
                        #     c_num = max(cnts, key=cv2.contourArea)
                        #     ((x, y), radius) = cv2.minEnclosingCircle(c)
                        #     M = cv2.moments(c_num)
                        #     center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        #     # only proceed if the radius meets a minimum size
                        #     if radius > 10:
                        #         # draw the circle and centroid on the frame,
                        #         # then update the list of tracked points
                        #         cv2.circle(frame, (int(x), int(y)), int(radius),
                        #                    (0, 255, 255), 2)
                        #         cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        #         pts.appendleft(center)
                    annotator.box_label([0, 0, 0, 0], annotate_result, color=colors(20, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / input_names[c] / f'{p.stem}.jpg', BGR=True)
                # LOGGER.info(f"FINAL {input_names[c]}  ({coords})")
                final_coords = {}
                for cate_c, coords_list in coords.items():
                    final_coords[cate_c] = [sum(x_i) / len(x_i) for x_i in zip(*coords_list)]
                    card_coords[cate_c] = final_coords[cate_c]
                # LOGGER.info(f"FINAL {input_names[c]}  ({final_coords}), cards {card_coords[cate_c]}")

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

            # Print time (inference-only)
            n_timer += 1
            s += f'Timer:{n_timer} '
            if n_timer == 150:
                for cat_i in cat_timer.keys():
                    if cat_timer[cat_i] >= 50:
                        if cat_i not in actual_inputs:
                            actual_inputs.append(cat_i)
                for i in actual_inputs:
                    if i not in card_order:
                        card_order.append(i)
                    act_i = card_coords[i.split(' ')[0]]
                    tarneeb_input[i.split(' ')[0]] = act_i
                cat_timer = {}
                s += f"/Category: {cat_timer} /Actual inputs: {actual_inputs} /cards: {card_coords} /tarneeb: {tarneeb_input}"

        # TODO: fix timer
        # annotator.box_label([1, 0, 0, 0], "timer " + str(n_timer), color=colors(20, True))
        if keyboard.is_pressed('e'):
            LOGGER.info('emptying inputs')
            tarneeb_input = {}
            actual_inputs = []
            card_order = []
            card_coords = {}
            cat_timer = {}
            final_coords = {}

        if keyboard.is_pressed('t'):
            LOGGER.info('starting Tarneeb')
            if not started:
                LOGGER.info('setting bids and trump card')
                tarneeb_data, started = start(tarneeb_input, tarneeb_data, tarneeb_p_names)
                if tarneeb_data != None:
                    for p in tarneeb_data['players']:
                        LOGGER.info(f"player '{p.name}' cards: {p.hand}")
                    LOGGER.info("")
                else:
                    tarneeb_data, started = start(tarneeb_input)

            if len(tarneeb_input) > 0:
                if tarneeb_rounds == 0:
                    tarneeb_data, tarneeb_rounds = play(tarneeb_data, tarneeb_input, card_order, tarneeb_rounds)
                else:
                    tarneeb_data, tarneeb_rounds = play(tarneeb_data, tarneeb_input, card_order, tarneeb_rounds)
                tarneeb_input = {}
                actual_inputs = []
                card_order = []
                card_coords = {}
                cat_timer = {}
                final_coords = {}
            else:
                LOGGER.info(f"each player must draw a card from their set")
                LOGGER.info('')

            if tarneeb_rounds == 13:
                print('tarneeb game {tarneeb_game_num} finished!')
                tarneeb_rounds = 0

                intiating_team = list(tarneeb_data['bid'].keys())[0]
                other_team = list(tarneeb_data['bid'].keys())[1]
                bid_init = tarneeb_data['bid'][intiating_team]
                init_points_earned = tarneeb_data['teamWins'][intiating_team]
                other_points_earned = tarneeb_data['teamWins'][other_team]
                print(f"the initiating team bid {bid_init}, and it earned {init_points_earned} Points")
                if bid_init == 13:
                    if init_points_earned == bid_init:
                        tarneeb_data['teamFinalPoints'][intiating_team] += (bid_init * 2)
                        tarneeb_data['teamFinalPoints'][other_team] += 0
                    elif init_points_earned < bid_init:
                        tarneeb_data['teamFinalPoints'][intiating_team] += -16
                        tarneeb_data['teamFinalPoints'][other_team] += other_points_earned
                elif init_points_earned == 13:
                    tarneeb_data['teamFinalPoints'][intiating_team] += 16
                    tarneeb_data['teamFinalPoints'][other_team] += 0
                elif init_points_earned >= bid_init:  # normal win
                    # print(f"{intiating_team} wins this game!")
                    tarneeb_data['teamFinalPoints'][intiating_team] += bid_init
                    tarneeb_data['teamFinalPoints'][other_team] += 0
                elif init_points_earned < bid_init:
                    tarneeb_data['teamFinalPoints'][intiating_team] += -bid_init
                    tarneeb_data['teamFinalPoints'][other_team] += other_points_earned
                else:
                    print('POINT CALCULATION ERROR')

                print(f"Game {tarneeb_game_num} Final Points so far: {tarneeb_data['teamFinalPoints']}")
                tarneeb_data['teamPoints'] = {'team 1': 0, 'team 2': 0}

                if tarneeb_data['teamFinalPoints'][intiating_team] >= 31:
                    print(f"Tarneeb Game Done! {intiating_team}({tarneeb_data['teamAssignments'][intiating_team]}) wins")
                    tarneeb_end = True
                elif tarneeb_data['teamFinalPoints'][other_team] >= 31:
                    print(f"Tarneeb Game Done! {other_team}({tarneeb_data['teamAssignments'][other_team]}) wins")
                    tarneeb_end = True

                tarneeb_game_num += 1


        if tarneeb_end:
            print('Thank you for playing!')
            break

        if n_timer == 150:
            if len(det):
                LOGGER.info(f"{s}{'' if len(det) else '(no detections) '}{dt[1].dt * 1E3:.1f}ms")
            n_timer = 0

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.25, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
