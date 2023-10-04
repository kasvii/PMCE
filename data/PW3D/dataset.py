import os
import os.path as osp
import numpy as np
import torch
import json
from pycocotools.coco import COCO

from core.config import cfg
from funcs_utils import save_obj
from smpl import SMPL
from coord_utils import cam2pixel, rigid_align, compute_error_accel
from _img_utils import split_into_chunks_mesh

class PW3D(torch.utils.data.Dataset):
    def __init__(self, data_split, args):
        dataset_name = 'PW3D'
        self.data_split = data_split
        self.data_path = osp.join(cfg.data_dir, dataset_name, 'pw3d_data')
        self.img_path = osp.join(cfg.data_dir, dataset_name, 'imageFiles')

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

        # H36M joint set
        self.human36_root_joint_idx = 0
        self.human36_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.human36_skeleton = (
            (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
            (2, 3), (0, 4), (4, 5), (5, 6))
        self.joint_regressor_human36 = torch.Tensor(self.mesh_model.joint_regressor_h36m)

        # COCO joint set
        self.coco_joint_num = 19  # 17 + 2, manually added pelvis and neck
        self.coco_joints_name = (
            'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        self.coco_flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        self.coco_skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15), #(5, 6), (11, 12))
            (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
        self.joint_regressor_coco = torch.Tensor(self.mesh_model.joint_regressor_coco)
        self.openpose_joints_name = ('Nose', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip',  'L_Knee', 'L_Ankle', 'R_Eye', 'L_Eye', 'R_Ear',  'L_Ear', 'Pelvis')

        input_joint_name = 'coco'
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(input_joint_name)

        self.img_paths, self.vid_names, self.img_shapes, self.poses, self.shapes, self.transes, self.genders, self.pred_pose2ds, self.joints_cam_coco, self.gt_joints_img_coco, self.joints_cam_h36m, self.features = self.load_data()  # self.video_indexes: 37 video, and indices of each video
        self.seqlen = cfg.DATASET.seqlen
        self.stride = cfg.DATASET.stride
        
        self.vid_indices = split_into_chunks_mesh(self.img_paths, self.seqlen, self.stride, self.poses, is_train=(set=='train'))

    def get_joint_setting(self, joint_category='human36'):
        joint_num = eval(f'self.{joint_category}_joint_num')
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')

        return joint_num, skeleton, flip_pairs

    def get_smpl_coord(self, pose_param, shape_param, trans_param, gender):
        pose, shape, trans = pose_param, shape_param, trans_param
        smpl_pose = torch.FloatTensor(pose).view(-1, 3);
        smpl_shape = torch.FloatTensor(shape).view(1, -1);
        # translation vector from smpl coordinate to 3dpw world coordinate
        smpl_trans = torch.FloatTensor(trans).view(-1, 3)

        smpl_pose = smpl_pose.view(1, -1)
        # get mesh and joint coordinates
        smpl_mesh_coord, smpl_joint_coord = self.mesh_model.layer[gender](smpl_pose, smpl_shape, smpl_trans)

        # incorporate face keypoints
        smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3);
        smpl_joint_coord = smpl_joint_coord.numpy().astype(np.float32).reshape(-1, 3)

        # meter -> milimeter
        smpl_mesh_coord *= 1000;
        smpl_joint_coord *= 1000;
        return smpl_mesh_coord, smpl_joint_coord

    def load_data(self):
        print('Load annotations of 3DPW ')
        db = COCO(osp.join(self.data_path, '3DPW_latest_' + self.data_split + '.json'))

        # get detected 2d pose
        if self.data_split == 'train':
            with open(osp.join(self.data_path, '3DPW_' + self.data_split +'_joint_coco_img_noise.json'), 'r') as f:
                joints_coco_img_noise = json.load(f)
        else:
            with open(osp.join(self.data_path,  f'vitpose_3dpw_{self.data_split}_output.json')) as f:
                pose2d_outputs = {}
                data = json.load(f)
                for item in data:
                    annot_id = str(item['annotation_id'])
                    pose2d_outputs[annot_id] = {'coco_joints': np.array(item['keypoints'], dtype=np.float32)[:, :3]}
                
        with open(osp.join(self.data_path, '3DPW_' + self.data_split +'_joint_coco_cam.json'), 'r') as f:
            coco_cam_joints = json.load(f)
            
        with open(osp.join(self.data_path, '3DPW_' + self.data_split +'_gt_joint_coco_img.json'), 'r') as f:
            gt_coco_img_joints = json.load(f)
            
        with open(osp.join(self.data_path, '3DPW_' + self.data_split +'_joint_h36m_cam.json'), 'r') as f:
            h36m_cam_joints = json.load(f)
            
        with open(osp.join(self.data_path, '3DPW_' + self.data_split +'_img_feat.json'), 'r') as f:
            raw_img_feats = json.load(f)

        img_paths, vid_names, img_shapes = [], [], []
        poses, shapes, transes, genders, pred_pose2ds = [], [], [], [], []
        joints_cam_coco, gt_joints_img_coco, joints_cam_h36m, img_feats = [], [], [], []
        for aid in db.anns.keys():
            aid = int(aid)
            ann = db.anns[aid]
            image_id = ann['image_id'] 
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']
            sequence_name = img['sequence']
            img_name = img['file_name']

            smpl_param = ann['smpl_param']
            pose_param = np.array(smpl_param['pose'], dtype=np.float32)
            shape_param = np.array(smpl_param['shape'], dtype=np.float32)
            trans = np.array(smpl_param['trans'], dtype=np.float32)
            
            pid = ann['person_id']
            img_path = osp.join(str(pid), sequence_name, img_name) # person/video/img
            vid_name = sequence_name + str(pid) # 按人划分的videos
            
            seq_idx = str(sequence_name)
            img_idx = str(int(img_name[6:-4]))
            person_idx = str(int(pid))
            feat_idx = seq_idx + '_' + person_idx + '_' + img_idx
            
            joint_cam_coco = np.array(coco_cam_joints[seq_idx][img_idx][person_idx], dtype=np.float32)
            gt_joint_img_coco = np.array(gt_coco_img_joints[seq_idx][img_idx][person_idx], dtype=np.float32)
            joint_cam_h36m = np.array(h36m_cam_joints[seq_idx][img_idx][person_idx], dtype=np.float32)
            
            try:
                img_feat = np.array(raw_img_feats[feat_idx], dtype=np.float32)
            except KeyError:
                continue

            if self.data_split == 'train':
                custompose = np.array(joints_coco_img_noise[seq_idx][img_idx][person_idx])
            else:
                custompose = np.array(pose2d_outputs[str(aid)]['coco_joints'])
                custompose = self.add_pelvis_and_neck(custompose, self.coco_joints_name)
                
            img_paths.append(img_path)
            vid_names.append(vid_name)
            img_shapes.append(np.array((img_height, img_width), dtype=np.int32))
            poses.append(np.array(pose_param, dtype=np.float32))
            shapes.append(np.array(shape_param, dtype=np.float32))
            transes.append(np.array(trans, dtype=np.float32))
            genders.append(smpl_param['gender'])
            pred_pose2ds.append(np.array(custompose, dtype=np.float32))
            joints_cam_coco.append(np.array(joint_cam_coco, dtype=np.float32))
            gt_joints_img_coco.append(np.array(gt_joint_img_coco, dtype=np.float32))
            joints_cam_h36m.append(np.array(joint_cam_h36m, dtype=np.float32))
            img_feats.append(np.array(img_feat, dtype=np.float32))
            
        img_paths, vid_names, img_shapes= np.array(img_paths), np.array(vid_names), np.array(img_shapes)
        poses, shapes, transes, genders, pred_pose2ds = np.array(poses), np.array(shapes), np.array(transes), np.array(genders), np.array(pred_pose2ds)
        joints_cam_coco, gt_joints_img_coco, joints_cam_h36m = np.array(joints_cam_coco), np.array(gt_joints_img_coco), np.array(joints_cam_h36m)
        img_feats = np.array(img_feats)
        
        perm = np.argsort(img_paths) 
        img_paths, vid_names, img_shapes = img_paths[perm], vid_names[perm], img_shapes[perm]
        poses, shapes, transes, genders, pred_pose2ds = poses[perm], shapes[perm], transes[perm], genders[perm], pred_pose2ds[perm]
        joints_cam_coco, gt_joints_img_coco, joints_cam_h36m = joints_cam_coco[perm], gt_joints_img_coco[perm], joints_cam_h36m[perm]
        img_feats = img_feats[perm]

        return img_paths, vid_names, img_shapes, poses, shapes, transes, genders, pred_pose2ds, joints_cam_coco, gt_joints_img_coco, joints_cam_h36m, img_feats

    def add_pelvis_and_neck(self, joint_coord, joints_name, only_pelvis=False):
        lhip_idx = joints_name.index('L_Hip')
        rhip_idx = joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis = pelvis.reshape((1, -1))

        lshoulder_idx = joints_name.index('L_Shoulder')
        rshoulder_idx = joints_name.index('R_Shoulder')
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck = neck.reshape((1,-1))

        if only_pelvis:
            joint_coord = np.concatenate((joint_coord, pelvis))
        else:
            joint_coord = np.concatenate((joint_coord, pelvis, neck))
        return joint_coord
    
    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        return X / w * 2 - [1, h / w]

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, idx):
        start_index, end_index = self.vid_indices[idx]
        joint_imgs = []
        img_features = []
        for num in range(self.seqlen):
            if start_index == end_index:
                single_idx = start_index
            else:
                single_idx = start_index + num
            img_shape = self.img_shapes[single_idx]
            pose_param, shape_param, trans_param, gender = self.poses[single_idx].copy(), self.shapes[single_idx].copy(), self.transes[single_idx].copy(), self.genders[single_idx]
            joint_img_coco = self.pred_pose2ds[single_idx].copy()
            joint_cam_coco, gt_joint_img_coco, joint_cam_h36m = self.joints_cam_coco[single_idx].copy(), self.gt_joints_img_coco[single_idx].copy(), self.joints_cam_h36m[single_idx].copy()
            root_coor = joint_cam_h36m[:1].copy()
            joint_cam_coco = joint_cam_coco - root_coor
            joint_cam_h36m = joint_cam_h36m - joint_cam_h36m[:1]

            if cfg.DATASET.use_gt_input:
                joint_img_coco = gt_joint_img_coco
            
            img_feature = self.features[single_idx].copy()
               
            joint_img_coco = joint_img_coco[:, :2]
            joint_img_coco = self.normalize_screen_coordinates(joint_img_coco, w=img_shape[1], h=img_shape[0])
            joint_img_coco = np.array(joint_img_coco, dtype=np.float32)

            joint_imgs.append(joint_img_coco.reshape(1, len(joint_img_coco), 2))
            img_features.append(img_feature.reshape(1, len(img_feature)))
            if cfg.MODEL.name == 'PMCE':
                if num == int(self.seqlen / 2):
                    mesh_cam, joint_cam_smpl = self.get_smpl_coord(pose_param, shape_param, trans_param, gender)
                    mesh_cam = mesh_cam - root_coor
                    mesh_valid = np.ones((len(mesh_cam), 1), dtype=np.float32)
                    reg_joint_valid = np.ones((len(joint_cam_h36m), 1), dtype=np.float32)
                    lift_joint_valid = np.ones((len(joint_cam_coco), 1), dtype=np.float32)
                    
                    targets = {'mesh': mesh_cam / 1000, 'lift_pose3d': joint_cam_coco, 'reg_pose3d': joint_cam_h36m}
                    meta = {'mesh_valid': mesh_valid, 'lift_pose3d_valid': lift_joint_valid, 'reg_pose3d_valid': reg_joint_valid}

            elif cfg.MODEL.name == 'PoseEst' and num == int(self.seqlen / 2):
                posenet_joint_cam = joint_cam_coco
                joint_valid = np.ones((len(joint_img_coco), 1), dtype=np.float32)
        
        joint_imgs = np.concatenate(joint_imgs)
        img_features = np.concatenate(img_features)
        if cfg.MODEL.name == 'PMCE':
            inputs = {'pose2d': joint_imgs, 'img_feature': img_features}
            return inputs, targets, meta
        elif cfg.MODEL.name == 'PoseEst':
            return joint_imgs, posenet_joint_cam, joint_valid, img_features

    def compute_joint_err(self, pred_joint, target_joint):
        # root align joint, coco joint set
        pred_joint, target_joint = pred_joint - pred_joint[:, -2:-1, :], target_joint - target_joint[:, -2:-1, :]
        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error

    def compute_both_err(self, pred_mesh, target_mesh, pred_joint, target_joint):
        # root align joint
        pred_mesh, target_mesh = pred_mesh - pred_joint[:, :1, :], target_mesh - target_joint[:,:1, :]
        pred_joint, target_joint = pred_joint - pred_joint[:, :1, :], target_joint - target_joint[:, :1, :]

        pred_mesh, target_mesh = pred_mesh.detach().cpu().numpy(), target_mesh.detach().cpu().numpy()
        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        pred_joint, target_joint = pred_joint[:, self.human36_eval_joint, :], target_joint[:, self.human36_eval_joint,
                                                                              :]
        mesh_mean_error = np.power((np.power((pred_mesh - target_mesh), 2)).sum(axis=2), 0.5).mean()
        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error, mesh_mean_error

    def evaluate_joint(self, outs):
        print('Evaluation start...')
        annots = self.vid_indices
        assert len(annots) == len(outs)
        sample_num = len(outs)

        mpjpe = np.zeros((sample_num, self.coco_joint_num))  # pose error
        pa_mpjpe = np.zeros((sample_num, self.coco_joint_num))  # pose error
        
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
            joint_coord_out = joint_coord_out - joint_coord_out[-2:-1]
            joint_coord_gt = joint_coord_gt - joint_coord_gt[-2:-1]

            # pose error calculate
            mpjpe[n] = np.sqrt(np.sum((joint_coord_out - joint_coord_gt) ** 2, 1))
            
            seq_name = self.vid_names[mid_index]
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
        eval_summary = '\nCOCO MPJPE (mm)     >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        tot_err = np.mean(pa_mpjpe)
        eval_summary = 'COCO PA-MPJPE (mm)  >> tot: %.2f\n' % (tot_err)
        print(eval_summary)
        
        acc_error = acc_error_h36m / sample_num
        acc_eval_summary = 'COCO ACCEL (mm/s^2) >> tot: %.2f\n ' % (acc_error)
        print(acc_eval_summary)

    def evaluate(self, outs):
        print('Evaluation start...')
        annots = self.vid_indices
        assert len(annots) == len(outs)
        sample_num = len(outs)

        mpjpe_h36m = np.zeros((sample_num, len(self.human36_eval_joint))) # pose error
        pampjpe_h36m = np.zeros((sample_num, len(self.human36_eval_joint))) # pose error

        mpvpe = np.zeros((sample_num, self.smpl_vertex_num)) # mesh error

        pred_j3d, gt_j3d = [], []
        
        pred_j3ds_h36m = []  # acc error for each sequence
        gt_j3ds_h36m = []  # acc error for each sequence
        acc_error_h36m = 0.0
        last_seq_name = None

        for n in range(sample_num):
            out = outs[n]
            start_index, end_index = self.vid_indices[n]
            if start_index == end_index:
                mid_index = start_index
            else:
                mid_index = start_index + int(self.seqlen / 2)
            img_path = self.img_paths[mid_index]

            mesh_coord_out, mesh_coord_gt = out['mesh_coord'], out['mesh_coord_target']
            joint_coord_out, joint_coord_gt = np.dot(self.joint_regressor_smpl, mesh_coord_out), np.dot(self.joint_regressor_smpl, mesh_coord_gt)
            # root joint alignment
            coord_out_cam = np.concatenate((mesh_coord_out, joint_coord_out))
            coord_out_cam = coord_out_cam - coord_out_cam[self.smpl_vertex_num + self.smpl_root_joint_idx]
            coord_gt_cam = np.concatenate((mesh_coord_gt, joint_coord_gt))
            coord_gt_cam = coord_gt_cam - coord_gt_cam[self.smpl_vertex_num + self.smpl_root_joint_idx]

            # mesh error calculate
            mesh_coord_out = coord_out_cam[:self.smpl_vertex_num,:]
            mesh_coord_gt = coord_gt_cam[:self.smpl_vertex_num,:]
            mpvpe[n] = np.sqrt(np.sum((mesh_coord_out - mesh_coord_gt)**2,1))

            # pose error of h36m calculate
            pose_coord_out_h36m = np.dot(self.mesh_model.joint_regressor_h36m, mesh_coord_out)
            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.human36_root_joint_idx]
            pose_coord_out_h36m = pose_coord_out_h36m[self.human36_eval_joint, :]
            pose_coord_gt_h36m = np.dot(self.mesh_model.joint_regressor_h36m, mesh_coord_gt)
            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[self.human36_root_joint_idx]
            pose_coord_gt_h36m = pose_coord_gt_h36m[self.human36_eval_joint, :]

            # pose error of coco calculate
            lhip_idx = self.coco_joints_name.index('L_Hip')
            rhip_idx = self.coco_joints_name.index('R_Hip')
            pose_coord_out_coco = np.dot(self.mesh_model.joint_regressor_coco, mesh_coord_out)
            pelvis_out = (pose_coord_out_coco[lhip_idx, :] + pose_coord_out_coco[rhip_idx, :]) * 0.5
            pelvis_out = pelvis_out.reshape((1, -1))
            pose_coord_out_coco = pose_coord_out_coco - pelvis_out
            pose_coord_gt_coco = np.dot(self.mesh_model.joint_regressor_coco, mesh_coord_gt)
            pelvis_gt = (pose_coord_gt_coco[lhip_idx, :] + pose_coord_gt_coco[rhip_idx, :]) * 0.5
            pelvis_gt = pelvis_gt.reshape((1, -1))
            pose_coord_gt_coco = pose_coord_gt_coco - pelvis_gt

            pred_j3d.append(pose_coord_out_h36m); gt_j3d.append(pose_coord_gt_h36m)

            seq_name = self.vid_names[mid_index]
            if last_seq_name is not None and seq_name != last_seq_name:
                pred_j3ds = np.array(pred_j3ds_h36m)
                target_j3ds = np.array(gt_j3ds_h36m)
                accel_err = np.zeros((len(pred_j3ds,)))
                accel_err[1:-1] = compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)
                err = np.mean(np.array(accel_err))
                acc_error_h36m += err.copy() * len(pred_j3ds)
                pred_j3ds_h36m = [pose_coord_out_h36m.copy()]
                gt_j3ds_h36m = [pose_coord_gt_h36m.copy()]
            else:
                pred_j3ds_h36m.append(pose_coord_out_h36m.copy())
                gt_j3ds_h36m.append(pose_coord_gt_h36m.copy())
            last_seq_name = seq_name
            
            mpjpe_h36m[n] = np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m)**2,1))
            pose_coord_out_h36m_aligned = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m) # perform rigid 
            pampjpe_h36m[n] = np.sqrt(np.sum((pose_coord_out_h36m_aligned - pose_coord_gt_h36m)**2,1))
            
            vis = cfg.TEST.vis
            if vis and (n % 1000 == 0):
                mesh_to_save = mesh_coord_out / 1000
                img_name = img_path.split('/')[-1]
                vid_name = img_path.split('/')[-2]
                obj_path = osp.join(cfg.vis_dir, f'3dpw_{vid_name}_{img_name[:-4]}.obj')
                save_obj(mesh_to_save, self.mesh_model.face, obj_path)
        
        pred_j3ds = np.array(pred_j3ds_h36m)
        target_j3ds = np.array(gt_j3ds_h36m)
        accel_err = np.zeros((len(pred_j3ds,)))
        accel_err[1:-1] = compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)
        err = np.mean(np.array(accel_err))
        acc_error_h36m += err.copy() * len(pred_j3ds)

        tot_err = np.mean(mpjpe_h36m)
        eval_summary = '\nH36M MPJPE (mm)     >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        tot_err = np.mean(pampjpe_h36m)
        eval_summary = 'H36M PA-MPJPE (mm)  >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        # total mesh error
        tot_err = np.mean(mpvpe)
        eval_summary = 'MPVPE (mm)          >> tot: %.2f\n' % (tot_err)
        print(eval_summary)
        
        acc_error = acc_error_h36m / sample_num
        eval_summary = 'H36M ACCEL (mm/s^2) >> tot: %.2f\n ' % (acc_error)
        print(eval_summary)