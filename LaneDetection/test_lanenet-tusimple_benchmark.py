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
from dataset_minicity_test import MinicityDataset


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='path to TuSimple Benchmark dataset')
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


if __name__ == '__main__':
    args = init_args()

    '''Test config'''
    batch_size = 1
    num_workers = 1
    train_start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    # writer = SummaryWriter(log_dir='summary/lane-detect-%s-%s' % (train_start_time, args.tag))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        num_gpus = torch.cuda.device_count()
        batch_size *= num_gpus
        print(f"Let's use {num_gpus} GPU(s)!")
    else:
        device = torch.device("cpu")
        print("Let's use CPU")
    print(f"Batch size: {batch_size}")

    output_dir = f'./output/Test-{train_start_time}-{args.tag}'
    os.makedirs(output_dir, exist_ok=True)

    data_dir = args.data_dir

    test_set = MinicityDataset(data_dir)
    # test_set = TuSimpleDataset('/root/Projects/lane_detection/dataset/tusimple/test_set', phase='test_extend')
    # val_set = TuSimpleDataset('/root/Projects/lane_detection/dataset/tusimple/train_set', phase='val')

    num_test = len(test_set)

    testset_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataloaders = {'test': testset_loader}
    phase = 'test'

    print(f'Finished loading data from {data_dir}')

    '''Constant variables'''
    VGG_MEAN = np.array([103.939, 116.779, 123.68]).astype(np.float32)

    _, h, w = test_set[0]['input_tensor'].shape

    # for IPM (Inverse Perspective Mapping)
    src = np.float32([
        [0.35 * (w - 1), 0.34 * (h - 1)],
        [0.65 * (w - 1), 0.34 * (h - 1)],
        [0.0 * (w - 1), h - 1],
        [1.0 * (w - 1), h - 1]
    ])
    dst = np.float32([
        [0.0 * (w - 1), 0.0 * (h - 1)],
        [1.0 * (w - 1), 0.0 * (h - 1)],
        [0.4 * (w - 1), h - 1],
        [0.60 * (w - 1), h - 1]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    # y_start, y_stop and y_num are calculated according to TuSimple Benchmark's setting
    y_start = int(np.round(240 * h / 720.))
    y_stop = int(np.round(710 * h / 720.))
    y_num = 48
    y_sample = np.linspace(y_start, y_stop, y_num, dtype=np.int16)
    x_sample = np.full_like(y_sample, w // 2, dtype=np.float32)
    c_sample = np.ones_like(y_sample, dtype=np.float32)
    xyc_sample = np.vstack((x_sample, y_sample, c_sample))

    xyc_IPM = M.dot(xyc_sample).T

    y_IPM = np.round(xyc_IPM[:, 1] / xyc_IPM[:, 2]).astype(np.int16)

    '''Forward propagation'''
    with torch.no_grad():
        # Select NN architecture
        arch = args.arch.lower()
        if 'fcn' in arch:
            arch = 'lanenet.LaneNet_FCN_Res'
        elif 'enet' in arch:
            arch = 'lanenet.LaneNet_ENet'
        elif 'icnet' in arch:
            arch = 'lanenet.LaneNet_ICNet'
        else:
            raise ValueError(f"Unsupported architecture: {args.arch}")

        arch += '_1E2D' if args.dual_decoder else '_1E1D'
        print(f'Architecture: {arch}')
        net = eval(arch)()

        # Initialize weights if necessary
        # net.apply(init_weights)

        net = nn.DataParallel(net)
        net.to(device)
        net.eval()

        assert args.ckpt_path is not None, 'Checkpoint path must be provided.'

        # Load checkpoint
        checkpoint = torch.load(args.ckpt_path, map_location=device,weights_only=True)
        if 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            net.load_state_dict(checkpoint, strict=True)

        step = 0
        epoch = 1
        print()

        data_iter = {'test': iter(dataloaders['test'])}
        time_run_avg = 0
        time_fp_avg = 0
        time_clst_avg = 0
        time_fit_avg = 0
        time_ct = 0
        output_list = []

        for step in range(num_test):
            time_run = time.time()
            time_fp = time.time()

            '''Load dataset'''
            try:
                batch = next(data_iter[phase])
            except StopIteration:
                break

            input_batch = batch['input_tensor'].to(device)  # Shape: (n, c, h, w)
            raw_file_batch = batch['raw_file']
            path_batch = batch['path']

            # Forward pass
            embeddings, logit = net(input_batch)

            # Prediction
            pred_bin_batch = torch.argmax(logit, dim=1, keepdim=True)  # Shape: (n, 1, h, w)

            time_fp = time.time() - time_fp

            '''Clustering with sklearn MeanShift'''
            time_clst = time.time()
            try:
                pred_insts = cluster_embed(embeddings, pred_bin_batch, band_width=0.5)  # Shape: (n, h, w)
            except ValueError as e:
                print(f"Clustering error: {e}")
                print(f"Embeddings shape: {embeddings.shape}")
                print(f"Pred_bin_batch shape: {pred_bin_batch.shape}")
                continue  # Skip to the next batch
            time_clst = time.time() - time_clst

            '''Curve Fitting'''
            time_fit = time.time()
            for idx in range(input_batch.size(0)):
                input_rgb = input_batch[idx]  # Shape: (c, h, w)
                raw_file = raw_file_batch[idx]
                pred_inst = pred_insts[idx]   # Shape: (h, w)
                path = path_batch[idx]

                if args.ipm:
                    '''Fit Curve after IPM (Inverse Perspective Mapping)'''
                    pred_inst_np = pred_inst.cpu().numpy().astype('uint8')
                    pred_inst_IPM = cv2.warpPerspective(pred_inst_np, M, (w, h), flags=cv2.INTER_NEAREST)
                    pred_inst_IPM = torch.from_numpy(pred_inst_IPM).to(device)

                    curves_param = fit_lanes(pred_inst_IPM)
                    curves_pts_IPM = sample_from_IPMcurve(curves_param, pred_inst_IPM, y_IPM)

                    curves_pts_pred = []
                    for xy_IPM in curves_pts_IPM:  # For each lane in an image
                        n_pts, _ = xy_IPM.shape

                        c_IPM = np.ones((n_pts, 1))
                        xyc_IPM = np.hstack((xy_IPM, c_IPM))
                        xyc_pred = M_inv.dot(xyc_IPM.T).T

                        xy_pred = []
                        for pt in xyc_pred:
                            x = int(np.round(pt[0] / pt[2]))
                            y = int(np.round(pt[1] / pt[2]))
                            if 0 <= y < h and 0 <= x < w:
                                xy_pred.append([x, y])
                            else:
                                xy_pred.append([-2, y])

                        xy_pred = np.array(xy_pred, dtype=np.int32)
                        curves_pts_pred.append(xy_pred)
                else:
                    '''Directly fit curves on original images'''
                    curves_param = fit_lanes(pred_inst)
                    curves_pts_pred = sample_from_curve(curves_param, pred_inst, y_sample)

                '''Visualization'''
                curve_sample = np.zeros((h, w, 3), dtype=np.uint8)
                rgb = (input_rgb.cpu().numpy().transpose(1, 2, 0) * 255 + VGG_MEAN).astype(np.uint8)
                pred_bin_rgb = pred_bin_batch[idx].repeat(3, 1, 1).cpu().numpy().transpose(1, 2, 0).astype(np.uint8) * 255
                pred_inst_rgb = pred_inst.repeat(3, 1, 1).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)

                for i in np.unique(pred_inst_rgb[:, :, 0]):
                    if i == 0:
                        continue
                    mask = pred_inst_rgb[:, :, 0] == i
                    pred_inst_rgb[mask] = get_color(i)

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
                    os.makedirs(output_seq_dir, exist_ok=True)

                    cv2.imwrite(os.path.join(output_seq_dir, f'{frame}_input.jpg'), rgb)
                    cv2.imwrite(os.path.join(output_seq_dir, f'{frame}_bin_pred.jpg'), pred_bin_rgb)
                    cv2.imwrite(os.path.join(output_seq_dir, f'{frame}_inst_pred.jpg'), pred_inst_rgb)
                    cv2.imwrite(os.path.join(output_seq_dir, f'{frame}_align.jpg'), rgb_align)

                if args.ipm:
                    curve_sample_IPM = np.zeros((h, w, 3), dtype=np.uint8)
                    rgb_IPM = cv2.warpPerspective(rgb, M, (w, h), flags=cv2.INTER_LINEAR)
                    pred_inst_IPM_rgb = cv2.warpPerspective(pred_inst_np, M, (w, h), flags=cv2.INTER_NEAREST)
                    pred_inst_IPM_rgb = np.stack([pred_inst_IPM_rgb]*3, axis=2).astype(np.uint8) * 40

                if args.show:
                    if args.ipm:
                        '''For IPM image'''
                        for inst in curves_pts_IPM:
                            # Filter valid points
                            valid_mask = np.logical_and(0 <= inst[:, 0], inst[:, 0] < w)
                            valid_mask = np.logical_and(valid_mask, inst[:, 1] >= y_start)
                            valid_mask = np.logical_and(valid_mask, inst[:, 1] <= y_stop)
                            inst = inst[valid_mask]

                            if inst.size == 0:
                                continue

                            pts = inst.astype(np.int32)
                            curve_sample_IPM[pts[:, 1], pts[:, 0]] = (0, 0, 255)
                            rgb_IPM[pts[:, 1], pts[:, 0]] = (0, 0, 255)
                            pred_inst_IPM_rgb[pts[:, 1], pts[:, 0]] = (0, 0, 255)

                            cv2.polylines(curve_sample_IPM, [pts], False, (0, 0, 255), 2)
                            cv2.polylines(rgb_IPM, [pts], False, (0, 0, 255), 2)
                            cv2.polylines(pred_inst_IPM_rgb, [pts], False, (0, 0, 255), 2)
                        cv2.imshow('input_IPM', rgb_IPM)
                        cv2.imshow('inst_pred_IPM', pred_inst_IPM_rgb)
                        cv2.imshow('curve_IPM', curve_sample_IPM)

                    else:
                        '''For front-face view image'''
                        for inst in curves_pts_pred:
                            if inst.ndim == 2:
                                valid_mask = inst[:, 0] != -2
                                inst = inst[valid_mask]

                                if inst.size == 0:
                                    continue

                                pts = inst.astype(np.int32)
                                curve_sample[pts[:, 1], pts[:, 0]] = (0, 0, 255)
                                rgb[pts[:, 1], pts[:, 0]] = (0, 0, 255)
                                pred_bin_rgb[pts[:, 1], pts[:, 0]] = (0, 0, 255)
                                pred_inst_rgb[pts[:, 1], pts[:, 0]] = (0, 0, 255)

                                rgb = np.ascontiguousarray(rgb)
                                pred_bin_rgb = np.ascontiguousarray(pred_bin_rgb)
                                pred_inst_rgb = np.ascontiguousarray(pred_inst_rgb)
                                cv2.polylines(rgb, [pts], False, (0, 0, 255), 2)
                                cv2.polylines(pred_bin_rgb, [pts], False, (0, 0, 255), 2)
                                cv2.polylines(pred_inst_rgb, [pts], False, (0, 0, 255), 2)

                        cv2.imshow('inst_pred', pred_inst_rgb)
                        cv2.imshow('curve', curve_sample)
                        cv2.imshow('align', rgb_align)
                        # cv2.imshow('input', rgb)
                        # cv2.imshow('bin_pred', pred_bin_rgb)

                    cv2.waitKey(0)  # Non-blocking; change to 0 for blocking if needed

                time_fit = time.time() - time_fit
                time_run = time.time() - time_run

                # Generate Json file to be evaluated by TuSimple Benchmark official eval script
                json_entry = generate_json_entry(curves_pts_pred, y_sample, raw_file, (h, w), time_run)
                output_list.append(json_entry)

                # Update running averages
                if time_ct > 0:
                    time_run_avg = (time_ct * time_run_avg + time_run) / (time_ct + 1)
                    time_fp_avg = (time_ct * time_fp_avg + time_fp) / (time_ct + 1)
                    time_clst_avg = (time_ct * time_clst_avg + time_clst) / (time_ct + 1)
                    time_fit_avg = (time_ct * time_fit_avg + time_fit) / (time_ct + 1)
                else:
                    time_run_avg = time_run
                    time_fp_avg = time_fp
                    time_clst_avg = time_clst
                    time_fit_avg = time_fit
                time_ct += 1

                if step % 50 == 0 and step != 0:
                    time_ct = 0

                print(f"{train_start_time}  {args.tag}  Epoch:{epoch}  Step:{step}  "
                      f"Time:{time_run*1000:.1f}ms  "
                      f"time_run_avg:{time_run_avg*1000:.1f}ms  "
                      f"time_fp_avg:{time_fp_avg*1000:.1f}ms  "
                      f"time_clst_avg:{time_clst_avg*1000:.1f}ms  "
                      f"time_fit_avg:{time_fit_avg*1000:.1f}ms  "
                      f"fps_avg:{int(1/(time_run_avg + 1e-9))}")

                '''Write to Tensorboard Summary'''
                # Uncomment and configure TensorBoard if needed
                # num_images = 3
                # inputs_images = (input_batch + VGG_MEAN)[:num_images, [2, 1, 0], :, :]  # .byte()
                # writer.add_images('image', inputs_images, step)
                # writer.add_images('Bin Pred', pred_bin_batch[:num_images], step)
                # embedding_img = F.normalize(embeddings[:num_images], 1, 1) / 2. + 0.5
                # writer.add_images('Embedding', embedding_img, step)

        # Save predictions to JSON
        output_json_path = f'output/test_pred-{train_start_time}-{args.tag}.json'
        with open(output_json_path, 'w') as f:
            for item in output_list:
                json.dump(item, f)
                f.write('\n')

        print(f"Saved predictions to {output_json_path}")

