import os
import os.path as osp
import numpy as np
import torch
import json
import math
import transforms3d
from pycocotools.coco import COCO

from Human36M.noise_stats import error_distribution
from core.config import cfg
from noise_utils import synthesize_pose
from smpl import SMPL
from coord_utils import cam2pixel, process_bbox, get_bbox, compute_error_accel, rigid_align
from aug_utils import j2d_processing, affine_transform, transform_joint_to_other_db
from _img_utils import split_into_chunks_mesh, split_into_chunks_pose
import joblib
from _kp_utils import convert_kps


class MPII3D(torch.utils.data.Dataset):
    def __init__(self, data_split, args):
        dataset_name = 'MPII3D'
        if data_split == 'test':
            data_split = 'val'
        self.data_split = data_split
        self.debug = args.debug
        self.data_path = osp.join(cfg.data_dir, dataset_name, 'mpii3d_data')
        self.smpl_param_path = osp.join(self.data_path, 'MPI-INF-3DHP_SMPL_NeuralAnnot.json')

        self.fitting_thr = 3.0

        # MuCo joint set
        self.mpii3d_joint_num = 17
        self.mpii3d_joints_name = (
        'Head', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee',
        'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Torso', 'Nose')
        self.mpii3d_root_joint_idx = self.mpii3d_joints_name.index('Pelvis')

        # SMPL joint set
        self.mesh_model = SMPL()
        self.smpl_root_joint_idx = self.mesh_model.root_joint_idx
        self.face_kps_vertex = self.mesh_model.face_kps_vertex
        self.smpl_vertex_num = 6890
        self.smpl_joint_num = 24
        self.smpl_flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        self.smpl_skeleton = (
            (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
            (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))
        self.joint_regressor_smpl = self.mesh_model.layer['neutral'].th_J_regressor

        # h36m skeleton
        self.human36_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.human36_joint_num = 17
        self.human36_joints_name = (
        'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.human36_skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
        self.human36_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.human36_root_joint_idx = self.human36_joints_name.index('Pelvis')
        self.human36_error_distribution = self.get_stat()
        self.joint_regressor_human36 = torch.Tensor(self.mesh_model.joint_regressor_h36m)

        # COCO joint set
        self.coco_joint_num = 19  # 17 + 2, manually added pelvis and neck
        self.coco_joints_name = (
            'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        self.coco_flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        self.coco_skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15),  # (5, 6), #(11, 12),
            (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
        self.coco_root_joint_idx = self.coco_joints_name.index('Pelvis')
        self.joint_regressor_coco = torch.Tensor(self.mesh_model.joint_regressor_coco)

        self.input_joint_name = cfg.DATASET.input_joint_set
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(self.input_joint_name)
        if self.data_split == 'train':
            self.img_paths, self.img_shapes, self.poses, self.shapes, self.transes,\
            self.joints_cam_coco, self.gt_joints_img_coco, self.joints_cam_h36m, \
            self.cam_params_focal, self.cam_params_princpt, self.cam_params_R, self.cam_params_t, self.img_feats, self.pred_pose2ds, self.joint_valids = self.load_data_train()
        else:
            self.img_paths, self.img_shapes, self.joints_cam, \
            self.img_feats, self.pred_pose2ds = self.load_data_val()
                
        self.seqlen = cfg.DATASET.seqlen
        self.stride = 16 if self.data_split == 'train' else 1

        if self.data_split == 'train':
            self.vid_indices = split_into_chunks_mesh(self.img_paths, self.seqlen, self.stride, self.poses, is_train=(set=='train'))
        else:
            self.vid_indices = split_into_chunks_pose(self.img_paths, self.seqlen, self.stride, is_train=(set=='train'))
        
    def get_stat(self):
        ordered_stats = []
        for joint in self.human36_joints_name:
            item = list(filter(lambda stat: stat['Joint'] == joint, error_distribution))[0]
            ordered_stats.append(item)

        return ordered_stats

    def generate_syn_error(self):
        noise = np.zeros((self.human36_joint_num, 2), dtype=np.float32)
        weight = np.zeros(self.human36_joint_num, dtype=np.float32)
        for i, ed in enumerate(self.human36_error_distribution):
            noise[i, 0] = np.random.normal(loc=ed['mean'][0], scale=ed['std'][0])
            noise[i, 1] = np.random.normal(loc=ed['mean'][1], scale=ed['std'][1])
            weight[i] = ed['weight']

        prob = np.random.uniform(low=0.0, high=1.0, size=self.human36_joint_num)
        weight = (weight > prob)
        noise = noise * weight[:, None]

        return noise

    def get_joint_setting(self, joint_category='human36'):
        joint_num = eval(f'self.{joint_category}_joint_num')
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')

        return joint_num, skeleton, flip_pairs

    def load_data_train(self):
        print('Load annotations of MPI-INF-3DHP ')
        db = COCO(osp.join(self.data_path, 'MPI-INF-3DHP.json'))
        
        with open(self.smpl_param_path) as f:
            smpl_params = json.load(f)
            
        with open(osp.join(self.data_path, 'MPI-INF-3DHP_camera.json')) as f:
            cam_params = json.load(f)
            
        with open(osp.join(self.data_path, 'MPII3D_' + self.data_split +'_joint_coco_cam.json'), 'r') as f:
            coco_cam_joints = json.load(f)
            
        with open(osp.join(self.data_path, 'MPII3D_' + self.data_split +'_gt_joint_coco_img.json'), 'r') as f:
            gt_coco_img_joints = json.load(f)
            
        with open(osp.join(self.data_path, 'MPII3D_' + self.data_split +'_joint_h36m_cam.json'), 'r') as f:
            h36m_cam_joints = json.load(f)
            
        with open(osp.join(self.data_path, 'MPII3D_' + self.data_split +'_joint_coco_img_noise.json'), 'r') as f:
            joints_coco_img_noise = json.load(f)

        feat_file = osp.join(self.data_path, 'mpii3d_train_scale12_db.pt')
        if osp.isfile(feat_file):
            img_features = {}
            feat_db = joblib.load(feat_file)
            for idx in range(len(feat_db['img_name'])):
                img_name = feat_db['img_name'][idx]
                img_features[img_name] = {'features': np.array(feat_db['features'][idx], dtype=np.float32)}
            
        img_paths, img_shapes = [], []
        poses, shapes, transes, genders = [], [], [], []
        joints_cam_coco, gt_joints_img_coco, joints_cam_h36m, img_feats = [], [], [], []
        cam_params_focal, cam_params_princpt, cam_params_R, cam_params_t = [], [], [], []
        img_feats, pred_pose2ds, joint_valids = [], [], []
        skip_feat_num = 0
        for aid in db.anns.keys():
            aid = int(aid)
            ann = db.anns[aid]
            image_id = ann['image_id'] 
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            sub_idx = int(img['subject_idx'])
            seq_idx = int(img['seq_idx'])
            vid_idx = int(img['cam_idx'])
            frame_idx = int(img['frame_idx'])
            
            img_name = osp.join('data/mpii_3d', 'S'+str(sub_idx), 'Seq'+str(seq_idx), 'video_'+str(vid_idx), str(frame_idx).zfill(6)+'.jpg')
            try:
                img_feat = img_features[img_name]['features']
            except KeyError:
                skip_feat_num += 1
                continue
            
            try:
                smpl_param = smpl_params[str(sub_idx)][str(seq_idx)][str(frame_idx)]
                pose, shape, trans = np.array(smpl_param['pose']), np.array(smpl_param['shape']), np.array(smpl_param['trans'])
                sum = pose.sum() + shape.sum() + trans.sum()
                if np.isnan(sum):
                    continue
            except KeyError:
                continue
            
            pose_param = np.array(smpl_param['pose'], dtype=np.float32)
            shape_param = np.array(smpl_param['shape'], dtype=np.float32)
            trans = np.array(smpl_param['trans'], dtype=np.float32)
            
            cam_param = cam_params[str(sub_idx)][str(seq_idx)][str(vid_idx)]
            img_height, img_width = cam_param['img_shape'][0], cam_param['img_shape'][1]
            cam_param_focal = cam_param['focal']
            cam_param_princpt = cam_param['princpt']
            cam_param_R = cam_param['R']
            cam_param_t = cam_param['t']
            
            img_path = self.data_path + '/MPI_INF_3DHP/' + 'S'+str(sub_idx) + '/' + 'Seq' + str(seq_idx) + '/imageFrames/' +'video_'+str(vid_idx) + '/' + str(frame_idx).zfill(6)+'.jpg'
            
            sub_idx = str(sub_idx)
            seq_idx = str(seq_idx)
            vid_idx = str(vid_idx)
            frame_idx = str(frame_idx)
            
             # regress h36m, coco joints
            joint_cam_coco = np.array(coco_cam_joints[sub_idx][seq_idx][vid_idx][frame_idx], dtype=np.float32)
            gt_joint_img_coco = np.array(gt_coco_img_joints[sub_idx][seq_idx][vid_idx][frame_idx], dtype=np.float32)
            joint_cam_h36m = np.array(h36m_cam_joints[sub_idx][seq_idx][vid_idx][frame_idx], dtype=np.float32)

            img_feat = np.array(img_feat, dtype=np.float32)
            
            joint_coco_img_noise = np.array(joints_coco_img_noise[sub_idx][seq_idx][vid_idx][frame_idx], dtype=np.float32)
            joint_valid = np.ones((len(joint_coco_img_noise), 1), dtype=np.float32)
 
            img_paths.append(img_path)
            img_shapes.append(np.array((img_height, img_width), dtype=np.int32))
            poses.append(np.array(pose_param, dtype=np.float32))
            shapes.append(np.array(shape_param, dtype=np.float32))
            transes.append(np.array(trans, dtype=np.float32))
            joints_cam_coco.append(np.array(joint_cam_coco, dtype=np.float32))
            gt_joints_img_coco.append(np.array(gt_joint_img_coco, dtype=np.float32))
            joints_cam_h36m.append(np.array(joint_cam_h36m, dtype=np.float32))
            img_feats.append(np.array(img_feat, dtype=np.float32))
            cam_params_focal.append(np.array(cam_param_focal, dtype=np.float32))
            cam_params_princpt.append(np.array(cam_param_princpt, dtype=np.float32))
            cam_params_R.append(np.array(cam_param_R, dtype=np.float32))
            cam_params_t.append(np.array(cam_param_t, dtype=np.float32))
            pred_pose2ds.append(np.array(joint_coco_img_noise, dtype=np.float32))
            joint_valids.append(np.array(joint_valid, dtype=np.float32))
            
        
        img_paths, img_shapes, poses, shapes, transes = np.array(img_paths), np.array(img_shapes), np.array(poses), np.array(shapes), np.array(transes)
        joints_cam_coco, gt_joints_img_coco, joints_cam_h36m = np.array(joints_cam_coco), np.array(gt_joints_img_coco), np.array(joints_cam_h36m)
        cam_params_focal, cam_params_princpt, cam_params_R, cam_params_t = np.array(cam_params_focal), np.array(cam_params_princpt), np.array(cam_params_R), np.array(cam_params_t)
        img_feats, pred_pose2ds, joint_valids = np.array(img_feats), np.array(pred_pose2ds), np.array(joint_valids)
        
        perm = np.argsort(img_paths) 
        img_paths, img_shapes, poses, shapes, transes = img_paths[perm], img_shapes[perm], poses[perm], shapes[perm], transes[perm]
        joints_cam_coco, gt_joints_img_coco, joints_cam_h36m = joints_cam_coco[perm], gt_joints_img_coco[perm], joints_cam_h36m[perm]
        cam_params_focal, cam_params_princpt, cam_params_R, cam_params_t = cam_params_focal[perm], cam_params_princpt[perm], cam_params_R[perm], cam_params_t[perm]
        img_feats, pred_pose2ds, joint_valids = img_feats[perm], pred_pose2ds[perm], joint_valids[perm]
        print('skip img feat is ', skip_feat_num)
        return img_paths, img_shapes, poses, shapes, transes,\
               joints_cam_coco, gt_joints_img_coco, joints_cam_h36m, \
               cam_params_focal, cam_params_princpt, cam_params_R, cam_params_t, img_feats, pred_pose2ds, joint_valids

    def load_data_val(self):
        print('Load annotations of MPI-INF-3DHP ')
        db_file = self.data_path + '/mpii3d_' + self.data_split + '_scale12_db.pt'
        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')
        
        with open(osp.join(self.data_path,  f'vitpose_mpii3d_{self.data_split}_output.json')) as f:
            pose2d_outputs = {}
            data = json.load(f)
            for item in data:
                annot_id = str(item['image_name'])
                pose2d_outputs[annot_id] = {'coco_joints': np.array(item['keypoints'], dtype=np.float32)[:, :3]}
            
        img_paths, img_shapes = [], []
        joints_cam, img_feats, pred_pose2ds = [], [], []
        for idx in range(len(db['img_name'])):
            img_name = db['img_name'][idx]
            img_feat = db['features'][idx]
            joint_cam = db['joints3D'][idx]
            joint_cam = convert_kps(joint_cam, "spin", "mpii3d_test").reshape((-1, 3))
            joint_cam = transform_joint_to_other_db(joint_cam, self.mpii3d_joints_name, self.human36_joints_name)
            joint_cam = joint_cam * 1000
            
            pred_pose2d = pose2d_outputs[str(img_name)]['coco_joints']
            pred_pose2d = self.add_pelvis_and_neck(pred_pose2d)
              
            img_paths.append(img_name)
            img_shapes.append(np.array((2048, 2048), dtype=np.int32))
            joints_cam.append(np.array(joint_cam, dtype=np.float32))
            img_feats.append(np.array(img_feat, dtype=np.float32))
            pred_pose2ds.append(np.array(pred_pose2d, dtype=np.float32))
        
        img_paths, img_shapes = np.array(img_paths), np.array(img_shapes)
        joints_cam = np.array(joints_cam)
        img_feats, pred_pose2ds = np.array(img_feats), np.array(pred_pose2ds)
        
        perm = np.argsort(img_paths) 
        img_paths, img_shapes = img_paths[perm], img_shapes[perm]
        joints_cam = joints_cam[perm]
        img_feats, pred_pose2ds = img_feats[perm], pred_pose2ds[perm]

        return img_paths, img_shapes, joints_cam, img_feats, pred_pose2ds
               
    def get_smpl_coord(self, pose_param, shape_param, trans_param, gender_param, cam_param_R, cam_param_t):
        pose, shape, trans, gender = pose_param, shape_param, trans_param, gender_param
        # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        smpl_pose = torch.FloatTensor(pose).view(-1, 3)
        smpl_shape = torch.FloatTensor(shape).view(1, -1)
        # translation vector from smpl coordinate to h36m world coordinate
        trans = np.array(trans, dtype=np.float32).reshape(3)
        # camera rotation and translation
        R, t = np.array(cam_param_R,dtype=np.float32).reshape(3, 3), np.array(cam_param_t,dtype=np.float32).reshape(3)

        # change to mean shape if beta is too far from it
        smpl_shape[(smpl_shape.abs() > 3).any(dim=1)] = 0.

        # transform world coordinate to camera coordinate
        root_pose = smpl_pose[self.smpl_root_joint_idx, :].numpy()
        angle = np.linalg.norm(root_pose)
        root_pose = transforms3d.axangles.axangle2mat(root_pose / angle, angle)
        root_pose = np.dot(R, root_pose)
        axis, angle = transforms3d.axangles.mat2axangle(root_pose)
        root_pose = axis * angle
        smpl_pose[self.smpl_root_joint_idx] = torch.from_numpy(root_pose)

        smpl_pose = smpl_pose.view(1, -1)

        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.mesh_model.layer[gender](smpl_pose, smpl_shape)

        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3);
        smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1, 3)

        # compenstate rotation (translation from origin to root joint was not cancled)
        smpl_trans = np.array(trans, dtype=np.float32).reshape(
            3)  # translation vector from smpl coordinate to h36m world coordinate
        smpl_trans = np.dot(R, smpl_trans[:, None]).reshape(1, 3) + t.reshape(1, 3) / 1000
        root_joint_coord = smpl_joint_coord[self.smpl_root_joint_idx].reshape(1, 3)
        smpl_trans = smpl_trans - root_joint_coord + np.dot(R, root_joint_coord.transpose(1, 0)).transpose(1, 0)

        # translation
        smpl_mesh_coord += smpl_trans; smpl_joint_coord += smpl_trans

        # meter -> milimeter
        smpl_mesh_coord *= 1000; smpl_joint_coord *= 1000;

        return smpl_mesh_coord, smpl_joint_coord

    def add_pelvis_and_neck(self, joint_coord):
        lhip_idx = self.coco_joints_name.index('L_Hip')
        rhip_idx = self.coco_joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis = pelvis.reshape((1, -1))

        lshoulder_idx = self.coco_joints_name.index('L_Shoulder')
        rshoulder_idx = self.coco_joints_name.index('R_Shoulder')
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck = neck.reshape((1, -1))

        joint_coord = np.concatenate((joint_coord, pelvis, neck))
        return joint_coord

    def get_joints_from_mesh(self, mesh, joint_set_name, cam_param):
        joint_coord_cam = None
        if joint_set_name == 'human36':
            joint_coord_cam = np.dot(self.joint_regressor_h36m, mesh)
        elif joint_set_name == 'coco':
            joint_coord_cam = np.dot(self.joint_regressor_coco, mesh)
            joint_coord_cam = self.add_pelvis_and_neck(joint_coord_cam)
        # projection
        f, c = np.array(cam_param['focal'], dtype=np.float32), np.array(cam_param['princpt'], dtype=np.float32)
        joint_coord_img = cam2pixel(joint_coord_cam, f, c)

        joint_coord_img[:, 2] = 1
        return joint_coord_cam, joint_coord_img

    def get_fitting_error(self, bbox, coco_from_dataset, coco_from_smpl, coco_joint_valid):
        bbox = process_bbox(bbox.copy(), aspect_ratio=1.0)
        coco_from_smpl_xy1 = np.concatenate((coco_from_smpl[:,:2], np.ones_like(coco_from_smpl[:,0:1])),1)
        coco_from_smpl, _ = j2d_processing(coco_from_smpl_xy1, (64, 64), bbox, 0, 0, None)
        coco_from_dataset_xy1 = np.concatenate((coco_from_dataset[:,:2], np.ones_like(coco_from_smpl[:,0:1])),1)
        coco_from_dataset, trans = j2d_processing(coco_from_dataset_xy1, (64, 64), bbox, 0, 0, None)

        # mask joint coordinates
        coco_joint = coco_from_dataset[:,:2][np.tile(coco_joint_valid,(1,2))==1].reshape(-1,2)
        coco_from_smpl = coco_from_smpl[:,:2][np.tile(coco_joint_valid,(1,2))==1].reshape(-1,2)

        error = np.sqrt(np.sum((coco_joint - coco_from_smpl)**2,1)).mean()
        return error

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, idx):
        if self.data_split == 'train':
            return self.get_single_item(idx)
        else:
            return self.get_single_item_val(idx)
    
    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        return X / w * 2 - [1, h / w]
    
    def get_single_item(self, idx):
        start_index, end_index = self.vid_indices[idx]
        joint_imgs = []
        img_features = []
        for num in range(self.seqlen):
            if start_index == end_index:
                single_idx = start_index
            else:
                single_idx = start_index + num
        
            img_shape = self.img_shapes[single_idx]
            pose_param, shape_param, trans_param, gender = self.poses[single_idx].copy(), self.shapes[single_idx].copy(), self.transes[single_idx].copy(), 'neutral'
            cam_param_R, cam_param_t = self.cam_params_R[single_idx].copy(), self.cam_params_t[single_idx].copy()
            joint_img_coco = self.pred_pose2ds[single_idx].copy()
            img_path = self.img_paths[single_idx].copy()
            img_feature = self.img_feats[single_idx].copy()

            joint_cam_coco, gt_joint_img_coco, joint_cam_h36m = self.joints_cam_coco[single_idx].copy(), self.gt_joints_img_coco[single_idx].copy(), self.joints_cam_h36m[single_idx].copy()
            root_coor = joint_cam_h36m[:1].copy()
            joint_cam_coco = joint_cam_coco - root_coor
            joint_cam_h36m = joint_cam_h36m - joint_cam_h36m[:1]

            if cfg.DATASET.use_gt_input:
                joint_img_coco = gt_joint_img_coco

            tight_bbox = get_bbox(joint_img_coco)
            bbox = process_bbox(tight_bbox.copy())
            joint_img_coco = joint_img_coco[:, :2]
            joint_img_coco = self.normalize_screen_coordinates(joint_img_coco, w=img_shape[1], h=img_shape[0])
            joint_img_coco = np.array(joint_img_coco, dtype=np.float32)
            
            joint_imgs.append(joint_img_coco.reshape(1, len(joint_img_coco), 2))
            img_features.append(img_feature.reshape(1, 2048))
            
            if cfg.MODEL.name == 'PMCE':
                if num == int(self.seqlen / 2):
                    # default valid
                    mesh_cam, joint_cam_smpl = self.get_smpl_coord(pose_param, shape_param, trans_param, gender, cam_param_R, cam_param_t)
                    # root relative camera coordinate
                    mesh_cam = mesh_cam - root_coor
                        
                    mesh_valid = np.ones((len(mesh_cam), 1), dtype=np.float32)
                    reg_joint_valid = np.ones((len(joint_cam_h36m), 1), dtype=np.float32)
                    lift_joint_valid = np.ones((len(joint_cam_coco), 1), dtype=np.float32)
                    
                    if self.data_split == 'train':
                        error = self.get_fitting_error(tight_bbox, self.pred_pose2ds[single_idx][:17], gt_joint_img_coco[:17], self.joint_valids[single_idx][:17])
                        if error > self.fitting_thr:
                            mesh_valid[:], reg_joint_valid[:], lift_joint_valid[:] = 0, 0, 0

                    targets = {'mesh': mesh_cam / 1000, 'lift_pose3d': joint_cam_coco, 'reg_pose3d': joint_cam_h36m}
                    meta = {'mesh_valid': mesh_valid, 'lift_pose3d_valid': lift_joint_valid, 'reg_pose3d_valid': reg_joint_valid}

            elif cfg.MODEL.name == 'PoseEst' and num == int(self.seqlen / 2):
                # default valid
                posenet_joint_cam = joint_cam_coco
                joint_valid = np.ones((len(joint_img_coco), 1), dtype=np.float32)
                # compute fitting error
                if self.data_split == 'train':
                    error = self.get_fitting_error(tight_bbox, self.pred_pose2ds[single_idx][:17], gt_joint_img_coco[:17], self.joint_valids[single_idx][:17])
                    if error > self.fitting_thr:
                        joint_valid[:, :] = 0
        
        joint_imgs = np.concatenate(joint_imgs)
        img_features = np.concatenate(img_features)
        if cfg.MODEL.name == 'PMCE':
            inputs = {'pose2d': joint_imgs, 'img_feature': img_features}
            return inputs, targets, meta
        
        elif cfg.MODEL.name == 'PoseEst':
            return joint_imgs, posenet_joint_cam, joint_valid, img_features

    def get_single_item_val(self, idx):
        start_index, end_index = self.vid_indices[idx]
        joint_imgs = []
        img_features = []
        for num in range(self.seqlen):
            if start_index == end_index:
                single_idx = start_index
            else:
                single_idx = start_index + num
        
            img_shape = self.img_shapes[single_idx]
            joint_img_coco = self.pred_pose2ds[single_idx].copy()
            img_path = self.img_paths[single_idx].copy()
            img_feature = self.img_feats[single_idx].copy()

            joint_cam = self.joints_cam[single_idx].copy()
            root_coor = joint_cam[0].copy()
            joint_cam = joint_cam - root_coor

            tight_bbox = get_bbox(joint_img_coco)
            bbox = process_bbox(tight_bbox.copy())
            joint_img_coco = joint_img_coco[:, :2]
            joint_img_coco = self.normalize_screen_coordinates(joint_img_coco, w=img_shape[1], h=img_shape[0])
            joint_img_coco = np.array(joint_img_coco, dtype=np.float32)
            
            joint_imgs.append(joint_img_coco.reshape(1, len(joint_img_coco), 2))
            img_features.append(img_feature.reshape(1, 2048))
            
            if cfg.MODEL.name == 'PMCE':
                if num == int(self.seqlen / 2):        
                    mesh_valid = np.zeros((6890, 1), dtype=np.float32)
                    reg_joint_valid = np.ones((len(joint_cam), 1), dtype=np.float32)
                    lift_joint_valid = np.zeros((len(joint_img_coco), 1), dtype=np.float32)
                    
                    targets = {'mesh': np.zeros((6890, 3), dtype=np.float32), 'lift_pose3d': np.zeros((len(joint_img_coco), 3), dtype=np.float32), 'reg_pose3d': joint_cam}
                    meta = {'mesh_valid': mesh_valid, 'lift_pose3d_valid': lift_joint_valid, 'reg_pose3d_valid': reg_joint_valid}

            elif cfg.MODEL.name == 'PoseEst' and num == int(self.seqlen / 2):
                # default valid
                posenet_joint_cam = np.zeros((len(joint_img_coco), 3), dtype=np.float32)
                joint_valid = np.zeros((len(joint_img_coco), 1), dtype=np.float32)
        
        joint_imgs = np.concatenate(joint_imgs)
        img_features = np.concatenate(img_features)
        if cfg.MODEL.name == 'PMCE':
            inputs = {'pose2d': joint_imgs, 'img_feature': img_features}
            return inputs, targets, meta
        
        elif cfg.MODEL.name == 'PoseEst':
            return joint_imgs, posenet_joint_cam, joint_valid, img_features
    
    def replace_joint_img(self, joint_img, bbox, trans):
        if self.input_joint_name == 'coco':
            joint_img_coco = joint_img
            if self.data_split == 'train':
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                pt1 = affine_transform(np.array([xmin, ymin]), trans)
                pt2 = affine_transform(np.array([xmax, ymin]), trans)
                pt3 = affine_transform(np.array([xmax, ymax]), trans)
                area = math.sqrt(pow(pt2[0] - pt1[0], 2) + pow(pt2[1] - pt1[1], 2)) * math.sqrt(
                    pow(pt3[0] - pt2[0], 2) + pow(pt3[1] - pt2[1], 2))
                joint_img_coco[:17, :] = synthesize_pose(joint_img_coco[:17, :], area, num_overlap=0)
                return joint_img_coco

        elif self.input_joint_name == 'human36':
            joint_img_h36m = joint_img
            if self.data_split == 'train':
                joint_syn_error = (self.generate_syn_error() / 256) * np.array(
                    [cfg.MODEL.input_shape[1], cfg.MODEL.input_shape[0]], dtype=np.float32)
                joint_img_h36m = joint_img_h36m[:, :2] + joint_syn_error
                return joint_img_h36m

    def compute_joint_err(self, pred_joint, target_joint):
        # root align joint, coco joint set
        pred_joint, target_joint = pred_joint - pred_joint[:, -2:-1, :], target_joint - target_joint[:, -2:-1, :]

        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error

    def compute_both_err(self, pred_mesh, target_mesh, pred_joint, target_joint):
        # root align joint
        pred_joint, target_joint = pred_joint - pred_joint[:, :1, :], target_joint - target_joint[:, :1, :]

        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        mesh_mean_error = 0.
        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error, mesh_mean_error

    def evaluate(self, outs):
        print('Evaluation start...')
        annots = self.vid_indices
        assert len(annots) == len(outs)
        sample_num = len(outs)

        mpjpe = np.zeros((sample_num, self.human36_joint_num))  # pose error
        pa_mpjpe = np.zeros((sample_num, self.human36_joint_num))  # pose error
        
        pred_j3ds_h36m = []  # acc error for each sequence
        gt_j3ds_h36m = []  # acc error for each sequence
        acc_error_h36m = 0.0
        last_seq_name = None

        for n in range(sample_num):
            start_index, end_index = self.vid_indices[n]
            if start_index == end_index:
                mid_index = start_index
            else:
                mid_index = start_index + int(self.seqlen / 2)
            out = outs[n]

            joint_coord_out, joint_coord_gt = out['joint_coord'], out['joint_coord_target']
            # root joint alignment, coco joint set
            joint_coord_out = joint_coord_out - joint_coord_out[0]
            joint_coord_gt = joint_coord_gt - joint_coord_gt[0]

            # pose error calculate
            mpjpe[n] = np.sqrt(np.sum((joint_coord_out - joint_coord_gt) ** 2, 1))
            
            seq_name = self.img_paths[mid_index][:-11]
            if last_seq_name is not None and seq_name != last_seq_name:
                pred_j3ds = np.array(pred_j3ds_h36m)
                target_j3ds = np.array(gt_j3ds_h36m)
                accel_err = np.zeros((len(pred_j3ds,)))
                accel_err[1:-1] = compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)
                err = np.mean(np.array(accel_err))
                acc_error_h36m += err.copy() * len(pred_j3ds)
                pred_j3ds_h36m = [joint_coord_out.copy()]
                gt_j3ds_h36m = [joint_coord_gt.copy()]
            else:
                pred_j3ds_h36m.append(joint_coord_out.copy())
                gt_j3ds_h36m.append(joint_coord_gt.copy())
            last_seq_name = seq_name
            
            joint_coord_out_aligned = rigid_align(joint_coord_out, joint_coord_gt)
            pa_mpjpe[n] = np.sqrt(np.sum((joint_coord_out_aligned - joint_coord_gt) ** 2, 1))

        pred_j3ds = np.array(pred_j3ds_h36m)
        target_j3ds = np.array(gt_j3ds_h36m)
        accel_err = np.zeros((len(pred_j3ds,)))
        accel_err[1:-1] = compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)
        err = np.mean(np.array(accel_err))
        acc_error_h36m += err.copy() * len(pred_j3ds)
        
        tot_err = np.mean(mpjpe)
        eval_summary = '\nH36M MPJPE (mm)     >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        tot_err = np.mean(pa_mpjpe)
        eval_summary = 'H36M PA-MPJPE (mm)  >> tot: %.2f\n' % (tot_err)
        print(eval_summary)
        
        acc_error = acc_error_h36m / sample_num
        acc_eval_summary = 'H36M ACCEL (mm/s^2) >> tot: %.2f\n ' % (acc_error)
        print(acc_eval_summary)