import os
import time
import argparse
import ujson as json
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

import model.lanenet as lanenet
from model.utils import cluster_embed, fit_lanes, sample_from_curve, sample_from_IPMcurve, generate_json_entry, get_color
from flask import Flask, request
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create upload folder if it doesn't exist


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, help='path to parameter file (.pth)')
    parser.add_argument('--arch', type=str, default='fcn', help='network architecture type(default: FCN)')
    parser.add_argument('--dual_decoder', action='store_true', help='use seperate decoders for two branches')
    parser.add_argument('--ipm', action='store_true', help='whether to perform Inverse Projective Mapping(IPM) before curve fitting')
    parser.add_argument('--show', action='store_true', help='whether to show visualization images when testing')
    parser.add_argument('--save_img', action='store_true', help='whether to save visualization images when testing')
    parser.add_argument('--tag', type=str, help='tag to log details of experiments')

    return parser.parse_args()


def init_weights(model):
    if type(model) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        torch.nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.01)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'tally' not in request.form:
        return 'File or tally part missing', 400
    
    file = request.files['file']
    tally = request.form['tally']  # Get tally from the form data
    
    if file.filename == '':
        return 'No selected file', 400

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (512,288), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image -= VGG_MEAN
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).float() / 255
    time_run = time.time()
    time_fp = time.time()

    input_batch = image.unsqueeze(0)

    raw_file_batch = file_path
    path_batch = file_path

    input_batch = input_batch.to(device)
    print(input_batch.shape)
    # forward
    embeddings, logit = net(input_batch)

    pred_bin_batch = torch.argmax(logit, dim=1, keepdim=True)
    preds_bin_expand_batch = pred_bin_batch.view(pred_bin_batch.shape[0] * pred_bin_batch.shape[1] * pred_bin_batch.shape[2] * pred_bin_batch.shape[3])

    time_fp = time.time() - time_fp

    '''sklearn mean_shift'''
    time_clst = time.time()
    try:
        pred_insts = cluster_embed(embeddings, pred_bin_batch, band_width=0.5)
    except ValueError as e:
        print(f"Clustering error: {e}")
        print(embeddings.shape)
        print(pred_bin_batch.shape)
    time_clst = time.time() - time_clst

    '''Curve Fitting'''
    time_fit = time.time()

    input_rgb = input_batch[0]  # for each image in a batch
    raw_file = raw_file_batch
    pred_inst = pred_insts[0]
    path = path_batch
    if args.ipm:
        '''Fit Curve after IPM(Inverse Perspective Mapping)'''
        pred_inst_IPM = cv2.warpPerspective(pred_inst.cpu().numpy().astype('uint8'), M, (w, h),
                                            flags=cv2.INTER_NEAREST)
        pred_inst_IPM = torch.from_numpy(pred_inst_IPM)

        curves_param = fit_lanes(pred_inst_IPM)
        curves_pts_IPM = sample_from_IPMcurve(curves_param, pred_inst_IPM, y_IPM)
        
        curves_pts_pred = []
        for xy_IPM in curves_pts_IPM:  # for each lane in a image
            n, _ = xy_IPM.shape
        
            c_IPM = np.ones((n, 1))
            xyc_IPM = np.hstack((xy_IPM, c_IPM))
            xyc_pred = M_inv.dot(xyc_IPM.T).T
        
            xy_pred = []
            for pt in xyc_pred:
                x = np.round(pt[0] / pt[2]).astype(np.int32)
                y = np.round(pt[1] / pt[2]).astype(np.int32)
                if 0 <= y < h and 0 <= x < w:  # and pred_inst[y, x]
                    xy_pred.append([x,y])
                else:
                    xy_pred.append([-2, y])
        
            xy_pred = np.array(xy_pred, dtype=np.int32)
            curves_pts_pred.append(xy_pred)
    else:
        '''Directly fit curves on original images'''
        #print(pred_inst.shape) #(288.512)
        curves_param = fit_lanes(pred_inst)
        curves_pts_pred=sample_from_curve(curves_param,pred_inst, y_sample)

    '''Visualization'''
    curve_sample = np.zeros((h, w, 3), dtype=np.uint8)
    rgb = (input_rgb.cpu().numpy().transpose(1, 2, 0) * 255 + VGG_MEAN).astype(np.uint8)
    pred_bin_rgb = pred_bin_batch[0].repeat(3,1,1).cpu().numpy().transpose(1, 2, 0).astype(np.uint8) * 255
    # pred_inst_rgb = pred_inst.repeat(3,1,1).cpu().numpy().transpose(1, 2, 0).astype(np.uint8) * 40  # gray
    pred_inst_rgb = pred_inst.repeat(3, 1, 1).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)  # color
    
    for i in np.unique(pred_inst_rgb):
        if i == 0:
            continue
        index = np.where(pred_inst_rgb[:, :, 0] == i)
        pred_inst_rgb[index] = get_color(i)

    fg_mask = (pred_bin_rgb[:, :, 0] == 255).astype(np.uint8)
    bg_mask = (pred_bin_rgb[:, :, 0] == 0).astype(np.uint8)
    rgb_bg = cv2.bitwise_and(rgb, rgb, mask=bg_mask)

    rgb_fg = cv2.bitwise_and(rgb, rgb, mask=fg_mask)
    pred_inst_rgb_fg = cv2.bitwise_and(pred_inst_rgb, pred_inst_rgb, mask=fg_mask)
    fg_align = cv2.addWeighted(rgb_fg, 0.3, pred_inst_rgb_fg, 0.7, 0)
    rgb_align = rgb_bg + fg_align

    if args.save_img:
        clip, seq, frame = path.split('/')
        output_seq_dir = os.path.join(output_dir, seq)
        if os.path.exists(output_seq_dir) is False:
            os.makedirs(output_seq_dir, exist_ok=True)

        cv2.imwrite(os.path.join(output_seq_dir, f'{frame}_input.jpg'), rgb)
        cv2.imwrite(os.path.join(output_seq_dir, f'{frame}_bin_pred.jpg'), pred_bin_rgb)
        cv2.imwrite(os.path.join(output_seq_dir, f'{frame}_inst_pred.jpg'), pred_inst_rgb)
        cv2.imwrite(os.path.join(output_seq_dir, f'{frame}_align.jpg'), rgb_align)

    if args.show:
        
        '''for front-face view image'''
        for idx, inst in enumerate(curves_pts_pred):
            if inst.ndim == 2:
                index = np.nonzero(inst[:, 0] != -2)
                inst = inst[index]
        
                pts = inst.transpose((1, 0))
                curve_sample[pts[1], pts[0]] = (0, 0, 255)
                rgb[pts[1], pts[0]] = (0, 0, 255)
                pred_bin_rgb[pts[1], pts[0]] = (0, 0, 255)
                pred_inst_rgb[pts[1], pts[0]] = (0, 0, 255)
                '''
                print(rgb.shape)
                print(rgb.dtype)
                print(inst.shape)
                print(inst.astype(np.int32))
                '''
                rgb = np.ascontiguousarray(rgb)
                pred_bin_rgb = np.ascontiguousarray(pred_bin_rgb)
                pred_inst_rgb = np.ascontiguousarray(pred_inst_rgb)
                cv2.polylines(rgb, [inst.astype(np.int32)], False, (0, 0, 255), 2)
                cv2.polylines(pred_bin_rgb, [inst.astype(np.int32)], False, (0, 0, 255), 2)
                cv2.polylines(pred_inst_rgb, [inst.astype(np.int32)], False, (0, 0, 255), 2)
            cv2.imshow('inst_pred', pred_inst_rgb)
            cv2.imshow('curve', curve_sample)
            cv2.imshow('align', rgb_align)
            # cv2.imshow('input',rgb)
            # cv2.imshow('bin_pred', pred_bin_rgb)
    time_fit = time.time() - time_fit
    time_run = time.time() - time_run
    time_run_avg = 0
    time_fp_avg = 0
    time_clst_avg = 0
    time_fit_avg = 0
    time_ct = 0
    time_run_avg = (time_ct * time_run_avg + time_run) / (time_ct + 1)
    time_fp_avg = (time_ct * time_fp_avg + time_fp) / (time_ct + 1)
    time_clst_avg = (time_ct * time_clst_avg + time_clst) / (time_ct + 1)
    time_fit_avg = (time_ct * time_fit_avg + time_fit) / (time_ct + 1)
    time_ct += 1

    print('{}  {}  Epoch:{}  Step:{}  Time:{:5.1f}  '
            'time_run_avg:{:5.1f}  time_fp_avg:{:5.1f}  time_clst_avg:{:5.1f}  time_fit_avg:{:5.1f}  fps_avg:{:d}'
            .format(train_start_time, args.tag, epoch, step, time_run*1000,
                    time_run_avg*1000, time_fp_avg * 1000, time_clst_avg * 1000, time_fit_avg * 1000,
                    int(1/(time_run_avg + 1e-9))))
    return {'status': 'File saved', 'tally': tally}, 200

if __name__ == '__main__':
    args = init_args()

    '''Test config'''
    batch_size = 1
    num_workers = 1
    train_start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    # writer = SummaryWriter(log_dir='summary/lane-detect-%s-%s' % (train_start_time, args.tag))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        batch_size *= torch.cuda.device_count()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        device = torch.device("cpu")
        print("Let's use CPU")
    print("Batch size: %d" % batch_size)

    output_dir = './output/Test-%s-%s' % (train_start_time, args.tag)
    if os.path.exists(output_dir) is False:
        os.makedirs(output_dir)
    phase = 'test'

    '''Constant variables'''
    VGG_MEAN = np.array([103.939, 116.779, 123.68]).astype(np.float32)

    h = 288
    w = 512

    # for IPM (Inverse Projective Mapping)
    src = np.float32([[0.35 * (w - 1), 0.34 * (h - 1)], [0.65 * (w - 1), 0.34 * (h - 1)],
                      [0. * (w - 1), h - 1], [1. * (w - 1), h - 1]])
    dst = np.float32([[0. * (w - 1), 0. * (h - 1)], [1.0 * (w - 1), 0. * (h - 1)],
                      [0.4 * (w - 1), (h - 1)], [0.60 * (w - 1), (h - 1)]])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    
    # y_start, y_stop and y_num is calculated according to TuSimple Benchmark's setting
    y_start = np.round(240 * h / 720.)
    y_stop = np.round(710 * h / 720.)
    y_num = 48
    y_sample = np.linspace(y_start, y_stop, y_num, dtype=np.int16)
    x_sample = np.zeros_like(y_sample, dtype=np.float32) + w // 2
    c_sample = np.ones_like(y_sample, dtype=np.float32)
    xyc_sample = np.vstack((x_sample, y_sample, c_sample))

    xyc_IPM = M.dot(xyc_sample).T

    y_IPM = []
    for pt in xyc_IPM:
        y = np.round(pt[1] / pt[2])
        y_IPM.append(y)
    y_IPM = np.array(y_IPM)

    '''Forward propogation'''
    with torch.no_grad():
        # select NN architecture
        arch = args.arch
        if 'fcn' in arch.lower():
            arch = 'lanenet.LaneNet_FCN_Res'
        elif 'enet' in arch.lower():
            arch = 'lanenet.LaneNet_ENet'
        elif 'icnet' in arch.lower():
            arch = 'lanenet.LaneNet_ICNet'
        
        arch = arch + '_1E2D' if args.dual_decoder else arch + '_1E1D'
        print('Architecture:', arch)
        net = eval(arch)()

        # net = lanenet.LaneNet_ENet_1E1D()

        net = nn.DataParallel(net)
        net.to(device)
        net.eval()

        assert args.ckpt_path is not None, 'Checkpoint Error.'

        checkpoint = torch.load(args.ckpt_path, map_location=torch.device('cpu'), weights_only=True)
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)

        step = 0
        epoch = 1
        print()
  

        app.run(host='0.0.0.0', port=80)


