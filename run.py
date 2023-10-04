import os
import sys
import os.path as osp
sys.path.append('lib')
sys.path.append('smplpytorch')
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import cv2
import torch
import joblib
import shutil
import colorsys
import argparse
import random
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

import models
from core.config import cfg
from aug_utils import j2d_processing
from coord_utils import get_bbox, process_bbox
from funcs_utils import load_checkpoint, save_obj
from demo.renderer import Renderer
from smpl import SMPL
import joblib
from pathlib import Path

from lib.utils._dataset_demo import CropDataset, FeatureDataset
from lib.utils.demo_utils import (
    download_youtube_clip,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
)


BASE_DATA_DIR = 'data/base_data'
MIN_NUM_FRAMES = 25
random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    x, y, w, h = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:, 3]
    cx, cy, h = x + w / 2., y + h / 2., h
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam


def render(verts, cam, bbox, orig_height, orig_width, orig_img, mesh_face, color, mesh_filename):
    pred_verts, pred_cam, bbox = verts, cam[None, :], bbox[None, :]

    orig_cam = convert_crop_cam_to_orig_img(
        cam=pred_cam,
        bbox=bbox,
        img_width=orig_width,
        img_height=orig_height
    )

    # Setup renderer for visualization
    renderer = Renderer(mesh_face, resolution=(orig_width, orig_height), orig_img=True, wireframe=False)
    renederd_img = renderer.render(
        orig_img,
        pred_verts,
        cam=orig_cam[0],
        color=color,
        mesh_filename=mesh_filename,
        rotate=False
    )

    return renederd_img


def get_joint_setting(mesh_model, joint_category='coco'):
    joint_regressor, joint_num, skeleton = None, None, None
    if joint_category == 'coco':
        joint_regressor = mesh_model.joint_regressor_coco
        joint_num = 19  # add pelvis and neck
        skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15),  # (5, 6), #(11, 12),
            (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
        flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        model_chk_path = 'experiment/pretrained/mesh_vis.pth.tar'

    else:
        raise NotImplementedError(f"{joint_category}: unknown joint set category")

    J_regressor = torch.Tensor(joint_regressor)
    model = models.PMCE.get_model(num_joint=joint_num, embed_dim=256, depth=3) 
    checkpoint = load_checkpoint(load_dir=model_chk_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, joint_regressor, joint_num, skeleton, model_chk_path

def add_pelvis_and_neck(joint_coord):
    lhip_idx = 11 # joints_name.index('L_Hip')
    rhip_idx = 12 # joints_name.index('R_Hip')
    pelvis = (joint_coord[:, lhip_idx, :] + joint_coord[:, rhip_idx, :]) * 0.5
    pelvis = pelvis.reshape((joint_coord.shape[0], 1, -1))

    lshoulder_idx = 5 # joints_name.index('L_Shoulder')
    rshoulder_idx = 6 # joints_name.index('R_Shoulder')
    neck = (joint_coord[:, lshoulder_idx, :] + joint_coord[:, rshoulder_idx, :]) * 0.5
    neck = neck.reshape((joint_coord.shape[0], 1, -1))

    joint_coord = np.concatenate((joint_coord, pelvis, neck), axis=1)
    return joint_coord

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]

def optimize_cam_param(project_net, model, joint_input, proj_target_joint_img, img_features, bbox1, joint_regressor):
    joint_img = torch.Tensor(joint_input[None, :, :, :]).cuda()
    target_joint = torch.Tensor(proj_target_joint_img[None, :, :2]).cuda()
    img_feat = torch.Tensor(img_features).cuda()

    # get optimization settings for projection
    criterion = nn.L1Loss()
    optimizer = optim.Adam(project_net.parameters(), lr=0.1)

    # estimate mesh, pose
    model.eval()
    pred_mesh, _, __ = model(joint_img, img_feat)
    pred_3d_joint = torch.matmul(joint_regressor, pred_mesh)


    out = {}
    # assume batch=1
    project_net.train()
    for j in range(0, 300):
        # projection
        pred_2d_joint = project_net(pred_3d_joint.detach())

        loss = criterion(pred_2d_joint, target_joint[:, :17, :])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if j == 100:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.05
        if j == 200:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001

    out['mesh'] = pred_mesh[0].detach().cpu().numpy()
    out['cam_param'] = project_net.cam_param[0].detach().cpu().numpy()
    out['bbox'] = bbox1

    out['target'] = proj_target_joint_img

    return out


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    """ Prepare input video (images) """
    video_file = args.vid_file
    if video_file.startswith('https://www.youtube.com'):
        print(f"Donwloading YouTube video \'{video_file}\'")
        video_file = download_youtube_clip(video_file, '/tmp')
        if video_file is None:
            exit('Youtube url is not valid!')
        print(f"YouTube Video has been downloaded to {video_file}...")

    if not os.path.isfile(video_file):
        exit(f"Input video \'{video_file}\' does not exist!")

    output_path = osp.join('./output/demo_output', os.path.basename(video_file).replace('.mp4', ''))
    Path(output_path).mkdir(parents=True, exist_ok=True)
    image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)

    print(f"Input video number of frames {num_frames}\n")
    orig_height, orig_width = img_shape[:2]


    """ Run tracking """
    bbox_scale = 1.1    #
    # run multi object tracker
    mot = MPT(
        device=device,
        batch_size=args.tracker_batch_size,
        display=args.display,
        detector_type=args.detector,
        output_format='dict',
        yolo_img_size=args.yolo_img_size,
    )
    tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]
           
    """ Prepare ViTPose """ 
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)
    from mmpose.datasets import DatasetInfo
    pose_model = init_pose_model(
    args.pose_config, args.pose_checkpoint, device=device)

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)


    """ Get PMCE model """
    seq_len = 16
    virtual_crop_size = 500
    joint_set = args.joint_set
    mesh_model = SMPL()
    model, joint_regressor, joint_num, skeleton, ckt_name = get_joint_setting(mesh_model, joint_category=joint_set)
    joint_regressor = torch.Tensor(joint_regressor).to(device)
    
    model = model.to(device)
    model.eval()
    
    project_net = models.project_net.get_model(crop_size=virtual_crop_size).to(device)
    
    # Get feature_extractor
    from lib.models.spin import hmr
    hmr = hmr().to(device)
    checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
    hmr.load_state_dict(checkpoint['model'], strict=False)
    hmr.eval()

    """ Run PMCE on each person """
    
    print("\nRunning PMCE on each person tracklet...")
    running_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None
        bboxes = tracking_results[person_id]['bbox']
        frames = tracking_results[person_id]['frames']
        joint2ds = []
        
        for idx, frame_id in enumerate(frames):
            img_path = osp.join(image_folder, str(frame_id + 1).zfill(6) + '.jpg')
            persons = []
            person_info = {}
            person_info['bbox'] = bboxes[idx]
            person_info['bbox'][0] = person_info['bbox'][0] - person_info['bbox'][2] * 0.5
            person_info['bbox'][1] = person_info['bbox'][1] - person_info['bbox'][3] * 0.5
            persons.append(person_info)
            
            pose_results, _ = inference_top_down_pose_model(
                pose_model,
                img_path,
                persons,
                bbox_thr=None,
                format='xywh',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=False,
                outputs=None)
            
            joint2ds.append(pose_results[0]['keypoints'])
            
        joints2d = joint2ds
            
        # Prepare static image features
        dataset = CropDataset(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        joints2d = dataset.joints2d
        has_keypoints = True if joints2d is not None else False

        crop_dataloader = DataLoader(dataset, batch_size=256, num_workers=0)

        with torch.no_grad():
            feature_list = []
            norm_joints2d = []
            for i, batch in enumerate(crop_dataloader):
                if has_keypoints:
                    batch, nj2d = batch
                    nj2d = add_pelvis_and_neck(nj2d[:, :, :2])
                    nj2d = torch.Tensor(nj2d)
                    norm_joints2d.append(nj2d.reshape(-1, joint_num, 2))

                batch = batch.to(device)
                feature = hmr.feature_extractor(batch.reshape(-1,3,224,224))
                feature_list.append(feature.cpu())

            del batch
            
            feature_list = torch.cat(feature_list, dim=0)
            norm_joints2d = torch.cat(norm_joints2d, dim=0)

        # Encode temporal features and estimate 3D human mesh
        dataset = FeatureDataset(
            image_folder=image_folder,
            frames=frames,
            seq_len=seq_len,
        )
        dataset.feature_list = feature_list
        dataset.joint2d_list = norm_joints2d

        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)     #32
        
        with torch.no_grad():
            pred_cam, pred_mesh, pred_bbox = [], [], []

            for i, batch in enumerate(dataloader):
                img_features, nj2d = batch

                nj2d = nj2d[0]
                bbox = get_bbox(nj2d[seq_len//2])
                bbox1 = process_bbox(bbox, aspect_ratio=1.0, scale=1.25)
                proj_target_joint_img, trans = j2d_processing(nj2d[seq_len//2].numpy(), (virtual_crop_size, virtual_crop_size), bbox1, 0, 0, None)
                norm_joint2d = normalize_screen_coordinates(nj2d.numpy(), orig_width, orig_height)
                
                with torch.enable_grad():
                    out = optimize_cam_param(project_net, model, norm_joint2d, proj_target_joint_img, img_features, bbox1, joint_regressor)
                    
                pred_mesh.append(out['mesh'])
                pred_cam.append(out['cam_param'])
                pred_bbox.append(out['bbox'])

            del batch

        # ========= Save results to a pickle file ========= #
        pred_mesh = np.array(pred_mesh)
        pred_cam = np.array(pred_cam)
        pred_bbox = np.array(pred_bbox)
        
        output_dict = {
            'pred_cam': pred_cam,
            'mesh': pred_mesh,
            'bboxes': pred_bbox,
            'frame_ids': frames,
        }

        running_results[person_id] = output_dict

    del model

    if args.save_pkl:
        print(f"Saving output results to \'{os.path.join(output_path, 'pmce_output.pkl')}\'.")
        joblib.dump(running_results, os.path.join(output_path, "pmce_output.pkl"))

    """ Render results as a single video """
    output_img_folder = f'{image_folder}_output'
    input_img_folder = f'{image_folder}_input'
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(input_img_folder, exist_ok=True)

    print(f"\nRendering output video, writing frames to {output_img_folder}")
    # prepare results for rendering
    frame_results = prepare_rendering_results(running_results, num_frames)
    color = (1.0, 0.6059142480254321, 0.5)

    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])

    for frame_idx in tqdm(range(len(image_file_names))):
        img_fname = image_file_names[frame_idx]
        img = cv2.imread(img_fname)
        input_img = img.copy()
        if args.render_plain:
            img[:] = 0

        if args.sideview:
            side_img = np.zeros_like(img)

        for person_id, person_data in frame_results[frame_idx].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']
            frame_bbox = person_data['bbox']

            mesh_filename = None
            if args.save_obj:
                mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                Path(mesh_folder).mkdir(parents=True, exist_ok=True)
                mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

            mc = color
            
            img = render(frame_verts, frame_cam, frame_bbox, orig_height, orig_width, img, mesh_model.face, mc, mesh_filename)

            if args.sideview:
                side_img = render(frame_verts, frame_cam, frame_bbox, orig_height, orig_width, 
                    side_img, mesh_model.face, color=mc, angle=270, axis=[0,1,0])

        if args.sideview:
            img = np.concatenate([img, side_img], axis=1)

        # save output frames
        cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.jpg'), img)
        cv2.imwrite(os.path.join(input_img_folder, f'{frame_idx:06d}.jpg'), input_img)

        if args.display:
            cv2.imshow('Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if args.display:
        cv2.destroyAllWindows()

    """ Save rendered video """
    vid_name = os.path.basename(video_file)
    save_name = f'pmce_{vid_name.replace(".mp4", "")}_output.mp4'
    save_path = os.path.join(output_path, save_name)

    images_to_video(img_folder=output_img_folder, output_vid_file=save_path)
    images_to_video(img_folder=input_img_folder, output_vid_file=os.path.join(output_path, vid_name))
    print(f"Saving result video to {os.path.abspath(save_path)}")
    shutil.rmtree(output_img_folder)
    shutil.rmtree(input_img_folder)
    shutil.rmtree(image_folder)    


if __name__ == '__main__':
    """
    python ./run.py --vid_file demo/sample_video.mp4 --gpu 0
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str, default='sample_video.mp4', help='input video path or youtube link')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')
    
    parser.add_argument('--joint_set', type=str, default='coco', help='choose the topology of input 2D pose')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')
    
    parser.add_argument('--pose_config', default='ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py', help='Config file for detection')
    
    parser.add_argument('--pose_checkpoint', default='pose_detector/vitpose-h-multi-coco.pth', help='Checkpoint file')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--save_pkl', action='store_true',
                        help='save results to a pkl file')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    parser.add_argument('--gender', type=str, default='neutral',
                        help='set gender of people from (neutral, male, female)')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--render_plain', action='store_true',
                        help='render meshes on plain background')

    parser.add_argument('--gpu', type=int, default='0', help='gpu number')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    main(args)