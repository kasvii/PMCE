import os.path as osp
import numpy as np
import math
import torch
import json
import copy
import transforms3d
import scipy.sparse
import cv2
from pycocotools.coco import COCO

from core.config import cfg 
from noise_utils import synthesize_pose

from smpl import SMPL
from coord_utils import world2cam, cam2pixel, process_bbox, rigid_align, get_bbox
from aug_utils import affine_transform, j3d_processing, flip_2d_joint

from funcs_utils import save_obj
import joblib
from _img_utils import split_into_chunks_pose, split_into_chunks_mesh
from eval_utils import compute_error_accel

class Human36M(torch.utils.data.Dataset):
    def __init__(self, mode, args):
        dataset_name = 'Human36M'
        self.debug = args.debug
        self.data_split = mode
        self.data_path = osp.join(cfg.data_dir, dataset_name, 'h36m_data')
        self.img_dir = osp.join(self.data_path, 'images')
        self.annot_path = osp.join(self.data_path, 'annotations')
        self.subject_genders = {1: 'female', 5: 'female', 6: 'male', 7: 'female', 8: 'male', 9: 'male', 11: 'male'}
        self.protocol = 2
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                            'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog',
                            'WalkTogether']
        self.fitting_thr = 25  # milimeter

        # SMPL joint set
        self.mesh_model = SMPL()
        self.smpl_root_joint_idx = self.mesh_model.root_joint_idx
        self.smpl_face_kps_vertex = self.mesh_model.face_kps_vertex
        self.smpl_vertex_num = 6890
        self.smpl_joint_num = 24
        self.smpl_flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))
        self.smpl_skeleton = (
            (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
            (17, 19), (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 15))
        self.joint_regressor_smpl = self.mesh_model.layer['neutral'].th_J_regressor

        # H36M joint set
        self.human36_joint_num = 17
        self.human36_joints_name = (
        'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.human36_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.human36_skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
        self.human36_root_joint_idx = self.human36_joints_name.index('Pelvis')
        # self.human36_error_distribution = self.get_stat()
        self.human36_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.joint_regressor_human36 = self.mesh_model.joint_regressor_h36m

        # COCO joint set
        self.coco_joint_num = 19  # 17 + 2, manually added pelvis and neck
        self.coco_joints_name = (
            'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
            'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
        self.coco_flip_pairs = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
        self.coco_skeleton = (
            (1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13),
            (13, 15), #(5, 6), #(11, 12),
            (17, 11), (17, 12), (17, 18), (18, 5), (18, 6), (18, 0))
        self.joint_regressor_coco = self.mesh_model.joint_regressor_coco

        self.input_joint_name = cfg.DATASET.input_joint_set
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(self.input_joint_name)

        self.img_paths, self.img_names, self.img_ids, self.bboxs, self.img_hws, self.joint_imgs, self.joint_cams, self.joint_vises, self.poses, self.shapes, self.transes, self.features, self.cam_idxs, self.joints_cam_h36m, \
        self.cam_param_focals, self.cam_param_princpts, self.cam_param_Rs, self.cam_param_ts= self.load_data()
        if self.data_split == 'test':
            det_2d_data_path = osp.join(self.data_path, 'Human36M_test_cpn_joint_2d.json')
            self.datalist_pose2d_det, self.datalist_pose2d_det_name = self.load_pose2d_det(det_2d_data_path)
        elif self.data_split == 'train':
            if self.input_joint_name == 'human36':
                det_2d_data_path = osp.join(self.data_path, 'Human36M_train_cpn_joint_2d.json')
                self.datalist_pose2d_det_train, self.datalist_pose2d_det_name_train = self.load_pose2d_det(det_2d_data_path)
            elif self.input_joint_name == 'coco':
                det_2d_data_path = self.annot_path
                self.datalist_pose2d_det_train, self.datalist_pose2d_det_name_train = self.load_pose2d_det(det_2d_data_path)
        self.seqlen = cfg.DATASET.seqlen
        # self.stride = cfg.DATASET.stride if self.data_split == 'train' else 1
        if self.input_joint_name == 'human36':
            self.stride = cfg.DATASET.stride if self.data_split == 'train' else 1
        elif self.input_joint_name == 'coco':
            self.stride = 16 if self.data_split == 'train' else 1

        if cfg.MODEL.name == 'PoseEst':
            self.vid_indices = split_into_chunks_pose(self.img_names, self.seqlen, self.stride, is_train=(set=='train'))
        elif cfg.MODEL.name == 'PMCE':
            self.vid_indices = split_into_chunks_mesh(self.img_names, self.seqlen, self.stride, self.poses, is_train=(set=='train'))


    def load_pose2d_det(self, data_path, skip_list=[]):
        pose2d_det = []
        pose2d_det_name = []
        if self.input_joint_name == 'human36':
            with open(data_path) as f:
                data = json.load(f)
                for img_path, pose2d in data.items():
                    pose2d = np.array(pose2d, dtype=np.float32)
                    if img_path in skip_list:
                        continue
                    pose2d_det.append(pose2d)
                    pose2d_det_name.append(img_path)
            perm = np.argsort(pose2d_det_name)
            pose2d_det, pose2d_det_name = np.array(pose2d_det), np.array(pose2d_det_name)
            pose2d_det, pose2d_det_name = pose2d_det[perm], pose2d_det_name[perm]
            subsampling_ratio = self.get_subsampling_ratio()
            if subsampling_ratio != 1:
                new_pose2d_det = []
                new_pose2d_det_name = []
                num = 0
                for idx, item in enumerate(pose2d_det):
                    img_idx = int(pose2d_det_name[idx][-10:-4]) - 1
                    if img_idx % subsampling_ratio == 0:
                        new_pose2d_det.append(item)
                        new_pose2d_det_name.append(pose2d_det_name[idx])
            else:
                new_pose2d_det = pose2d_det
                new_pose2d_det_name = pose2d_det_name   
            new_pose2d_det, new_pose2d_det_name = np.array(new_pose2d_det), np.array(new_pose2d_det_name)
        elif self.input_joint_name == 'coco':
            subject_list = self.get_subject()
            new_pose2d_det = []
            new_pose2d_det_name = []
            joints = {}
            
            for subject in subject_list:
                with open(osp.join(data_path, 'Human36M_subject' + str(subject) + '_joint_coco_img_noise_neuralannot.json'), 'r') as f:
                    joints[str(subject)] = json.load(f)
                    
            for img_name in self.img_names:
                subject = str(int(img_name.split('_')[1]))
                action_idx = str(int(img_name.split('_')[3]))
                subaction_idx = str(int(img_name.split('_')[5]))
                cam_idx = str(int(img_name.split('_')[7]))
                frame_idx = str(int(img_name[-10:-4]) - 1) # frame_idx = img_idx - 1
                pose2d = joints[str(subject)][str(action_idx)][str(subaction_idx)][str(cam_idx)][str(frame_idx)]
                pose2d = np.array(pose2d, dtype=np.float32)
                new_pose2d_det.append(pose2d)
                new_pose2d_det_name.append(img_name)
                
            perm = np.argsort(new_pose2d_det_name)
            new_pose2d_det, new_pose2d_det_name = np.array(new_pose2d_det), np.array(new_pose2d_det_name)
            new_pose2d_det, new_pose2d_det_name = new_pose2d_det[perm], new_pose2d_det_name[perm]
        return new_pose2d_det, new_pose2d_det_name

    def get_joint_setting(self, joint_category='human36'):
        joint_num = eval(f'self.{joint_category}_joint_num')
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')

        return joint_num, skeleton, flip_pairs

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 2
        elif self.data_split == 'test':
            return 2
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            if self.protocol == 1:
                subject = [1, 5, 6, 7, 8, 9]
            elif self.protocol == 2:
                subject = [1, 5, 6, 7, 8]
        elif self.data_split == 'test':
            if self.protocol == 1:
                subject = [11]
            elif self.protocol == 2:
                subject = [9, 11]
        else:
            assert 0, print("Unknown subset")

        if self.debug:
            subject = subject[0:1]

        return subject

    def load_data(self):
        print('Load annotations of Human36M Protocol ' + str(self.protocol))
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()

        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        joints_h36m = {}
        smpl_params = {}

        db_file = osp.join(self.data_path, 'h36m_' + self.data_split + '_imgfeat_db_concat.pt')
        if osp.isfile(db_file):
            img_db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')
        img_feats = img_db['features']
        img_names = img_db['img_name']
        perm = np.argsort(img_names)
        img_feats, img_names = img_feats[perm], img_names[perm]
        with open(osp.join(self.data_path, 'Human36M_' + self.data_split + '_start_idx_tight.json'), 'r') as f:
            start_idx = json.load(f)
        feat_cnt = - sampling_ratio

        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'), 'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k, v in annot.items():
                    db.dataset[k] = v
            else:
                for k, v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'), 'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            if self.input_joint_name == 'human36':
                with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
                    joints[str(subject)] = json.load(f)
            elif self.input_joint_name == 'coco':
                with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_coco_cam_3d_neuralannot.json'), 'r') as f:
                    joints[str(subject)] = json.load(f)
                with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'), 'r') as f:
                    joints_h36m[str(subject)] = json.load(f)
            # smpl parameter load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_SMPL_NeuralAnnot.json'), 'r') as f:
                smpl_params[str(subject)] = json.load(f)
        db.createIndex()

        img_paths, image_names, img_ids, bboxs, img_hws = [], [], [], [], []
        joint_imgs, joint_cams, joint_vises, poses, shapes, transes = [], [], [], [], [], []
        joint_cams_h36m = []
        features, cam_idxs = [], []
        cam_param_focals, cam_param_princpts, cam_param_Rs, cam_param_ts = [], [], [], []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_name = img_path.split('/')[-1]

            # check subject and frame_idx
            frame_idx = img['frame_idx'];

            if frame_idx % sampling_ratio != 0:
                continue
            feat_cnt += sampling_ratio
            
            if img_name[:-12] == 's_11_act_02_subact_02_ca_0':
                continue

            # check smpl parameter exist
            subject = img['subject'];
            action_idx = img['action_idx'];

            subaction_idx = img['subaction_idx'];
            frame_idx = img['frame_idx'];

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject)][str(cam_idx)]
            R, t, f, c = np.array(cam_param['R'], dtype=np.float32), \
                         np.array(cam_param['t'], dtype=np.float32), \
                         np.array(cam_param['f'], dtype=np.float32), \
                         np.array(cam_param['c'], dtype=np.float32)
            
            try:
                smpl_param = smpl_params[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)]
            except KeyError:
                smpl_param = None

            if smpl_param is not None:
                pose_param = np.array(smpl_param['pose'], dtype=np.float32)
                shape_param = np.array(smpl_param['shape'], dtype=np.float32)
                trans_param = np.array(smpl_param['trans'], dtype=np.float32)
            else:
                pose_param = np.zeros(1, dtype=np.float32)
                shape_param = np.zeros(1, dtype=np.float32)
                trans_param = np.zeros(1, dtype=np.float32)

            bbox = process_bbox(np.array(ann['bbox'], dtype=np.float32))
            if bbox is None: continue
            
            # project world coordinate to cam, image coordinate space
            if self.input_joint_name == 'human36':
                joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
                joint_cam = world2cam(joint_world, R, t)
                joint_img = cam2pixel(joint_cam, f, c)
                joint_vis = np.ones((self.human36_joint_num, 1), dtype=np.float32)
            elif self.input_joint_name == 'coco':
                joint_cam = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(cam_idx)][str(frame_idx)], dtype=np.float32)
                joint_img = cam2pixel(joint_cam, f, c)
                joint_img[:, 2] = 1
                joint_vis = np.ones((self.coco_joint_num, 1), dtype=np.float32)
                
                joint_world_h36m = np.array(joints_h36m[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)],
                                       dtype=np.float32)
                joint_cam_h36m = world2cam(joint_world_h36m, R, t)

            if frame_idx == 0:
                feat_cnt = start_idx[str(subject)][str(action_idx)][str(subaction_idx)][str(cam_idx)]
            feat_img_name = img_names[feat_cnt].split('/')[-1]
            assert img_name == feat_img_name
            feature = np.array(img_feats[feat_cnt], dtype=np.float32)

            img_paths.append(img_path)
            image_names.append(img_name)
            img_ids.append(image_id)
            joint_vises.append(joint_vis)
            bboxs.append(np.array(bbox, dtype=np.float32))
            img_hws.append(np.array((img['height'], img['width']), dtype=np.int32))
            joint_imgs.append(np.array(joint_img, dtype=np.float32))
            joint_cams.append(np.array(joint_cam, dtype=np.float32))
            poses.append(pose_param)
            shapes.append(shape_param)
            transes.append(trans_param)
            features.append(feature)
            cam_idxs.append(cam_idx)
            cam_param_focals.append(f)
            cam_param_princpts.append(c)
            cam_param_Rs.append(R)
            cam_param_ts.append(t)
            if self.input_joint_name == 'coco':
                joint_cams_h36m.append(np.array(joint_cam_h36m, dtype=np.float32))

        img_paths, image_names, img_ids, bboxs, img_hws = np.array(img_paths), np.array(image_names), np.array(img_ids, dtype=np.int32), np.array(bboxs), np.array(img_hws)
        joint_imgs, joint_cams, joint_vises, poses, shapes, transes = np.array(joint_imgs), np.array(joint_cams), np.array(joint_vises), np.array(poses), np.array(shapes), np.array(transes)
        features, cam_idxs = np.array(features), np.array(cam_idxs)
        cam_param_focals, cam_param_princpts, cam_param_Rs, cam_param_ts = np.array(cam_param_focals), np.array(cam_param_princpts), np.array(cam_param_Rs), np.array(cam_param_ts)
        if self.input_joint_name == 'coco':
            joint_cams_h36m = np.array(joint_cams_h36m)
        
        return img_paths, image_names, img_ids, bboxs, img_hws, joint_imgs, joint_cams, joint_vises, poses, shapes, transes, features, cam_idxs, joint_cams_h36m, \
               cam_param_focals, cam_param_princpts, cam_param_Rs, cam_param_ts



    def get_smpl_coord(self, smpl_param, cam_param):
        pose, shape, trans, gender = smpl_param['pose'], smpl_param['shape'], smpl_param['trans'], smpl_param['gender']
        # smpl parameters (pose: 72 dimension, shape: 10 dimension)
        smpl_pose = torch.FloatTensor(pose).view(-1, 3)
        smpl_shape = torch.FloatTensor(shape).view(1, -1)
        # translation vector from smpl coordinate to h36m world coordinate
        trans = np.array(trans, dtype=np.float32).reshape(3)
        # camera rotation and translation
        R, t = np.array(cam_param['R'],dtype=np.float32).reshape(3, 3), np.array(cam_param['t'],dtype=np.float32).reshape(3)

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

    def get_fitting_error(self, h36m_joint, smpl_mesh):
        h36m_joint = h36m_joint - h36m_joint[self.human36_root_joint_idx,None,:] # root-relative

        h36m_from_smpl = np.dot(self.joint_regressor_human36, smpl_mesh)
        # translation alignment
        h36m_from_smpl = h36m_from_smpl - np.mean(h36m_from_smpl,0)[None,:] + np.mean(h36m_joint,0)[None,:]
        error = np.sqrt(np.sum((h36m_joint - h36m_from_smpl)**2,1)).mean()
        return error

    def get_coco_from_mesh(self, mesh_coord_cam, cam_param):
        # regress coco joints
        joint_coord_cam = np.dot(self.joint_regressor_coco, mesh_coord_cam)
        joint_coord_cam = self.add_pelvis_and_neck(joint_coord_cam)
        # projection
        f, c = cam_param['focal'], cam_param['princpt']
        joint_coord_img = cam2pixel(joint_coord_cam, f, c)

        joint_coord_img[:, 2] = 1
        return joint_coord_cam, joint_coord_img

    def add_pelvis_and_neck(self, joint_coord):
        lhip_idx = self.coco_joints_name.index('L_Hip')
        rhip_idx = self.coco_joints_name.index('R_Hip')
        pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
        pelvis = pelvis.reshape((1, -1))

        lshoulder_idx = self.coco_joints_name.index('L_Shoulder')
        rshoulder_idx = self.coco_joints_name.index('R_Shoulder')
        neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
        neck = neck.reshape((1,-1))

        joint_coord = np.concatenate((joint_coord, pelvis, neck))
        return joint_coord

    def __len__(self):
        return len(self.vid_indices)
    
    def __getitem__(self, idx):
        return self.get_single_item(idx)
    
    def get_sequence(self, start_index, end_index, data):
        if start_index != end_index:
            return list(range(start_index, end_index+1))
        else:
            final_data = []
            single_data = data[start_index]
            for i in range(self.seqlen):
                final_data.append(single_data)
            return final_data

    def get_single_item(self, idx):
        start_index, end_index = self.vid_indices[idx]
        joint_imgs = []
        img_features = []
        flip, rot = 0, 0
        for num in range(self.seqlen):
            if start_index == end_index:
                single_idx = start_index
            else:
                single_idx = start_index + num
            img_id, bbox, img_shape = self.img_ids[single_idx], self.bboxs[single_idx].copy(), self.img_hws[single_idx]
            image_name = self.img_names[single_idx]
            cam_param = {'R': self.cam_param_Rs[single_idx], 't': self.cam_param_ts[single_idx], 'focal': self.cam_param_focals[single_idx], 'princpt': self.cam_param_princpts[single_idx]}
            smpl_param = {'pose': self.poses[single_idx], 'shape': self.shapes[single_idx], 'trans': self.transes[single_idx], 'gender': 'neutral'}
            
            img_feature = self.features[single_idx].copy()
            img_name = self.img_names[single_idx]

            # h36m joints from datasets
            # joint_cam is PoseEst target
            if self.input_joint_name == 'human36':
                joint_cam_h36m, joint_img_h36m = self.joint_cams[single_idx].copy(), self.joint_imgs[single_idx].copy()
                root_coord = joint_cam_h36m[:1].copy()
                joint_cam_h36m = joint_cam_h36m - joint_cam_h36m[:1]
                joint_img, joint_cam = joint_img_h36m, joint_cam_h36m
            
            # coco joints from datasets
            elif self.input_joint_name == 'coco':
                joint_cam_h36m = self.joints_cam_h36m[single_idx].copy()
                root_coord = joint_cam_h36m[:1].copy()
                joint_cam_h36m = joint_cam_h36m - joint_cam_h36m[:1]
                joint_cam_coco, joint_img_coco = self.joint_cams[single_idx].copy(), self.joint_imgs[single_idx].copy()
                joint_cam_coco = joint_cam_coco - root_coord
                joint_img, joint_cam = joint_img_coco, joint_cam_coco
                
            # make new bbox
            tight_bbox = get_bbox(joint_img)
            bbox = process_bbox(tight_bbox.copy())
            if not cfg.DATASET.use_gt_input:
                joint_img = self.replace_joint_img_wo_bbox(single_idx, img_id, joint_img, img_name, tight_bbox, w=img_shape[1], h=img_shape[0])
            if flip:
                joint_img = flip_2d_joint(joint_img, img_shape[1], self.flip_pairs)
            joint_img = joint_img[:,:2]
            joint_img = self.normalize_screen_coordinates(joint_img, w=img_shape[1], h=img_shape[0])
            joint_img = np.array(joint_img, dtype=np.float32)
            joint_cam = j3d_processing(joint_cam, rot, flip, self.flip_pairs)

            joint_imgs.append(joint_img.reshape(1, len(joint_img), 2))
            img_features.append(img_feature.reshape(1, 2048))
            if cfg.MODEL.name == 'PMCE':
                if num == int(self.seqlen / 2):
                    # default valid
                    mesh_cam, joint_h36m_from_mesh = self.get_smpl_coord(smpl_param, cam_param)
                    # root relative camera coordinate
                    mesh_cam = mesh_cam - root_coord
                    # draw_nodes_nodes(joint_cam_h36m, mesh_cam)
                    mesh_valid = np.ones((len(mesh_cam), 1), dtype=np.float32)
                    reg_joint_valid = np.ones((len(joint_cam_h36m), 1), dtype=np.float32)
                    lift_joint_valid = np.ones((len(joint_cam), 1), dtype=np.float32)
                    # if fitted mesh is too far from h36m gt, discard it
                    error = self.get_fitting_error(joint_cam_h36m, mesh_cam)
                    if error > self.fitting_thr:
                        mesh_valid[:] = 0
                        if self.input_joint_name == 'coco':
                            lift_joint_valid[:] = 0
                    targets = {'mesh': mesh_cam / 1000, 'lift_pose3d': joint_cam, 'reg_pose3d': joint_cam_h36m}
                    meta = {'mesh_valid': mesh_valid, 'lift_pose3d_valid': lift_joint_valid, 'reg_pose3d_valid': reg_joint_valid}

            elif cfg.MODEL.name == 'PoseEst' and num == int(self.seqlen / 2):
                # default valid
                posenet_joint_cam = joint_cam
                joint_valid = np.ones((len(joint_cam), 1), dtype=np.float32)
        
        joint_imgs = np.concatenate(joint_imgs)
        img_features = np.concatenate(img_features)
        if cfg.MODEL.name == 'PMCE':
            inputs = {'pose2d': joint_imgs, 'img_feature': img_features}
            return inputs, targets, meta
        
        elif cfg.MODEL.name == 'PoseEst':
            return joint_imgs, posenet_joint_cam, joint_valid, img_features

    def normalize_screen_coordinates(self, X, w, h):
        assert X.shape[-1] == 2
        return X / w * 2 - [1, h / w]

    def replace_joint_img_wo_bbox(self, idx, img_id, joint_img, img_name, bbox, w, h):
        if self.input_joint_name == 'coco':
            joint_img_coco = joint_img
            if self.data_split == 'train':
                det_data = self.datalist_pose2d_det_train[idx]
                det_name = self.datalist_pose2d_det_name_train[idx]
                assert img_name == det_name, f"check: {img_name} / {det_name}"
                joint_img_coco = det_data[:, :2].copy()
                return joint_img_coco
            else:
                det_data = self.datalist_pose2d_det[idx]
                joint_img_coco = det_data[:, :2].copy()
                return joint_img_coco
        if self.input_joint_name == 'human36':
            joint_img_h36m = joint_img
            if self.data_split == 'train':
                det_data = self.datalist_pose2d_det_train[idx]
                det_name = self.datalist_pose2d_det_name_train[idx]
                assert img_name == det_name, f"check: {img_name} / {det_name}"
                joint_img_h36m = det_data[:, :2].copy()
                return joint_img_h36m
            else:
                det_data = self.datalist_pose2d_det[idx]
                det_name = self.datalist_pose2d_det_name[idx]
                assert img_name == det_name, f"check: {img_name} / {det_name}"
                joint_img_h36m = det_data[:, :2].copy()
                return joint_img_h36m
    
    def replace_joint_img(self, idx, img_id, joint_img, bbox, trans, img_name):
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
            else:
                joint_img_coco = self.datalist_pose2d_det[img_id]
                joint_img_coco = self.add_pelvis_and_neck(joint_img_coco)
                for i in range(self.coco_joint_num):
                    joint_img_coco[i, :2] = affine_transform(joint_img_coco[i, :2].copy(), trans)
                return joint_img_coco

        elif self.input_joint_name == 'human36':
            joint_img_h36m = joint_img
            if self.data_split == 'train':
                det_data = self.datalist_pose2d_det_train[idx]
                assert img_name == det_data['img_name'], f"check: {img_name} / {det_data['img_name']}"
                joint_img_h36m = det_data['pose2d'][:, :2].copy()
                for i in range(self.human36_joint_num):
                    joint_img_h36m[i, :2] = affine_transform(joint_img_h36m[i, :2].copy(), trans)
                return joint_img_h36m
            else:
                det_data = self.datalist_pose2d_det[idx]
                assert img_name == det_data['img_name'], f"check: {img_name} / {det_data['img_name']}"
                joint_img_h36m = det_data['pose2d'][:, :2].copy()
                for i in range(self.human36_joint_num):
                    joint_img_h36m[i, :2] = affine_transform(joint_img_h36m[i, :2].copy(), trans)
                return joint_img_h36m

    def compute_joint_err(self, pred_joint, target_joint):
        # root align joint
        pred_joint, target_joint = pred_joint - pred_joint[:, :1, :], target_joint - target_joint[:, :1, :]

        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        pred_joint, target_joint = pred_joint[:, self.human36_eval_joint, :], target_joint[:, self.human36_eval_joint, :]
        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error

    def compute_both_err(self, pred_mesh, target_mesh, pred_joint, target_joint):
        # root align joint
        pred_mesh, target_mesh = pred_mesh - pred_joint[:, :1, :], target_mesh - target_joint[:, :1, :]
        pred_joint, target_joint = pred_joint - pred_joint[:, :1, :], target_joint - target_joint[:, :1, :]

        pred_mesh, target_mesh = pred_mesh.detach().cpu().numpy(), target_mesh.detach().cpu().numpy()
        pred_joint, target_joint = pred_joint.detach().cpu().numpy(), target_joint.detach().cpu().numpy()

        pred_joint, target_joint = pred_joint[:, self.human36_eval_joint, :], target_joint[:, self.human36_eval_joint, :]
        mesh_mean_error = np.power((np.power((pred_mesh - target_mesh), 2)).sum(axis=2), 0.5).mean()
        joint_mean_error = np.power((np.power((pred_joint - target_joint), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error, mesh_mean_error

    def evaluate_joint(self, outs):
        print('Evaluation start...')
        annots = self.vid_indices
        assert len(annots) == len(outs)
        sample_num = len(annots)
        
        sample_num_new = 0
        for i in range(sample_num):
            start_index, end_index = self.vid_indices[i]
            if start_index == end_index:
                mid_index = start_index
            else:
                mid_index = start_index + int(self.seqlen / 2)
            cam_idx = self.cam_idxs[mid_index]
            if cam_idx != 4:
                continue
            sample_num_new += 1

        mpjpe = np.zeros((sample_num_new, len(self.human36_eval_joint)))
        pampjpe = np.zeros((sample_num_new, len(self.human36_eval_joint)))
        
        pred_j3ds_h36m = []  # acc error for each sequence
        gt_j3ds_h36m = []  # acc error for each sequence
        acc_error_h36m = 0.0
        last_seq_name = None
        
        i = 0
        for n in range(sample_num):
            start_index, end_index = self.vid_indices[n]
            if start_index == end_index:
                mid_index = start_index
            else:
                mid_index = start_index + int(self.seqlen / 2)
            out = outs[n]
            
            cam_idx = self.cam_idxs[mid_index]
            if cam_idx != 4:
                continue

            # render materials
            pose_coord_out, pose_coord_gt = out['joint_coord'], self.joint_cams[mid_index]

            # root joint alignment
            pose_coord_out, pose_coord_gt = pose_coord_out - pose_coord_out[:1], pose_coord_gt - pose_coord_gt[:1]
            # sample eval joitns
            pose_coord_out, pose_coord_gt = pose_coord_out[self.human36_eval_joint, :], pose_coord_gt[self.human36_eval_joint, :]

            # pose error calculate
            mpjpe[i] = np.sqrt(np.sum((pose_coord_out - pose_coord_gt) ** 2, 1))
            
            seq_name = self.img_names[mid_index][:-11]
            if last_seq_name is not None and seq_name != last_seq_name:
                pred_j3ds = np.array(pred_j3ds_h36m)
                target_j3ds = np.array(gt_j3ds_h36m)
                accel_err = np.zeros((len(pred_j3ds,)))
                accel_err[1:-1] = compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)
                err = np.mean(np.array(accel_err))
                acc_error_h36m += err.copy() * len(pred_j3ds)
                pred_j3ds_h36m = [pose_coord_out.copy()]
                gt_j3ds_h36m = [pose_coord_gt.copy()]
            else:
                pred_j3ds_h36m.append(pose_coord_out.copy())
                gt_j3ds_h36m.append(pose_coord_gt.copy())
            last_seq_name = seq_name
            
            # perform rigid alignment
            pose_coord_out = rigid_align(pose_coord_out, pose_coord_gt)
            pampjpe[i] = np.sqrt(np.sum((pose_coord_out - pose_coord_gt) ** 2, 1))
                
            i = i + 1

        pred_j3ds = np.array(pred_j3ds_h36m)
        target_j3ds = np.array(gt_j3ds_h36m)
        accel_err = np.zeros((len(pred_j3ds,)))
        accel_err[1:-1] = compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)
        err = np.mean(np.array(accel_err))
        acc_error_h36m += err.copy() * len(pred_j3ds)

        tot_err = np.mean(mpjpe)
        eval_summary = '\nH36M MPJPE (mm)     >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        tot_err = np.mean(pampjpe)
        eval_summary = 'H36M PA-MPJPE (mm)  >> tot: %.2f\n' % (tot_err)
        print(eval_summary)
        
        acc_error = acc_error_h36m / i
        acc_eval_summary = 'H36M ACCEL (mm/s^2) >> tot: %.2f\n ' % (acc_error)
        print(acc_eval_summary)

    def evaluate(self, outs):
        print('Evaluation start...')
        annots = self.vid_indices
        assert len(annots) == len(outs)
        sample_num = len(outs)

        sample_num_new = 0
        for i in range(sample_num):
            start_index, end_index = self.vid_indices[i]
            if start_index == end_index:
                mid_index = start_index
            else:
                mid_index = start_index + int(self.seqlen / 2)
            cam_idx = self.cam_idxs[mid_index]
            if cam_idx != 4:
                continue
            sample_num_new += 1

        # eval H36M joints
        pose_error_h36m = np.zeros((sample_num_new, len(self.human36_eval_joint)))  # pose error
        pose_error_action_h36m = [[] for _ in range(len(self.action_name))]  # pose error for each sequence
        pose_pa_error_h36m = np.zeros((sample_num_new, len(self.human36_eval_joint)))  # pose error
        pose_pa_error_action_h36m = [[] for _ in range(len(self.action_name))]  # pose error for each sequence
        
        pred_j3ds_h36m = []  # acc error for each sequence
        gt_j3ds_h36m = []  # acc error for each sequence
        acc_error_h36m = 0.0
        last_seq_name = None
        
        # eval SMPL joints and mesh vertices
        pose_error = np.zeros((sample_num_new, self.smpl_joint_num))  # pose error
        pose_error_action = [[] for _ in range(len(self.action_name))]  # pose error for each sequence
        mesh_error = np.zeros((sample_num_new, self.smpl_vertex_num))  # mesh error
        mesh_error_action = [[] for _ in range(len(self.action_name))]  # mesh error for each sequence

        n = 0
        for i in range(sample_num):
            start_index, end_index = self.vid_indices[i]
            if start_index == end_index:
                mid_index = start_index
            else:
                mid_index = start_index + int(self.seqlen / 2)
            out = outs[i]

            cam_idx = self.cam_idxs[mid_index]
            if cam_idx != 4:
                continue

            # render materials
            img_path = self.img_paths[mid_index]
            obj_name = '_'.join(img_path.split('/')[-2:])[:-4]

            # root joint alignment
            mesh_coord_out, mesh_coord_gt = out['mesh_coord'], out['mesh_coord_target']
            joint_coord_out, joint_coord_gt = np.dot(self.joint_regressor_smpl, mesh_coord_out), np.dot(self.joint_regressor_smpl, mesh_coord_gt)
            mesh_coord_out = mesh_coord_out - joint_coord_out[self.smpl_root_joint_idx:self.smpl_root_joint_idx+1]
            mesh_coord_gt = mesh_coord_gt - joint_coord_gt[self.smpl_root_joint_idx:self.smpl_root_joint_idx+1]
            pose_coord_out = joint_coord_out - joint_coord_out[self.smpl_root_joint_idx:self.smpl_root_joint_idx+1]
            pose_coord_gt = joint_coord_gt - joint_coord_gt[self.smpl_root_joint_idx:self.smpl_root_joint_idx+1]

            # pose error calculate
            pose_error[n] = np.sqrt(np.sum((pose_coord_out - pose_coord_gt) ** 2, 1))
            img_name = self.img_paths[mid_index]
            action_idx = int(img_name[img_name.find('act') + 4:img_name.find('act') + 6]) - 2
            pose_error_action[action_idx].append(pose_error[n].copy())

            # mesh error calculate
            mesh_error[n] = np.sqrt(np.sum((mesh_coord_out - mesh_coord_gt) ** 2, 1))
            img_name = self.img_paths[mid_index]
            action_idx = int(img_name[img_name.find('act') + 4:img_name.find('act') + 6]) - 2
            mesh_error_action[action_idx].append(mesh_error[n].copy())

            # pose error of h36m calculate
            pose_coord_out_h36m = np.dot(self.joint_regressor_human36, mesh_coord_out)
            pose_coord_out_h36m = pose_coord_out_h36m - pose_coord_out_h36m[self.human36_root_joint_idx]
            pose_coord_out_h36m = pose_coord_out_h36m[self.human36_eval_joint, :]
            pose_coord_gt_h36m = self.joint_cams[mid_index]
            pose_coord_gt_h36m = pose_coord_gt_h36m - pose_coord_gt_h36m[self.human36_root_joint_idx]
            pose_coord_gt_h36m = pose_coord_gt_h36m[self.human36_eval_joint, :]
            
            seq_name = self.img_names[mid_index][:-11]
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
                
            pose_error_h36m[n] = np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m) ** 2, 1))
            pose_coord_out_h36m = rigid_align(pose_coord_out_h36m, pose_coord_gt_h36m) # perform rigid alignment
            pose_pa_error_h36m[n] = np.sqrt(np.sum((pose_coord_out_h36m - pose_coord_gt_h36m) ** 2, 1))
            img_name = self.img_paths[mid_index]
            action_idx = int(img_name[img_name.find('act') + 4:img_name.find('act') + 6]) - 2
            pose_error_action_h36m[action_idx].append(pose_error_h36m[n].copy())
            pose_pa_error_action_h36m[action_idx].append(pose_pa_error_h36m[n].copy())

            vis = cfg.TEST.vis
            if vis and (n % 500 == 0):
                mesh_to_save = mesh_coord_out / 1000
                obj_path = osp.join(cfg.vis_dir, f'{obj_name}.obj')
                save_obj(mesh_to_save, self.mesh_model.face, obj_path)

            n = n + 1

        pred_j3ds = np.array(pred_j3ds_h36m)
        target_j3ds = np.array(gt_j3ds_h36m)
        accel_err = np.zeros((len(pred_j3ds,)))
        accel_err[1:-1] = compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)
        err = np.mean(np.array(accel_err))
        acc_error_h36m += err.copy() * len(pred_j3ds)
            
        # total pose error (H36M joint set)
        tot_err = np.mean(pose_error_h36m)
        eval_summary = '\nProtocol ' + str(self.protocol) + ' H36M MPJPE (mm)     >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        tot_err = np.mean(pose_pa_error_h36m)
        eval_summary = 'Protocol ' + str(self.protocol) + ' H36M PA-MPJPE (mm)  >> tot: %.2f\n' % (tot_err)
        print(eval_summary)

        # total mesh error
        tot_err = np.mean(mesh_error)
        eval_summary = 'Protocol ' + str(self.protocol) + ' MPVPE (mm)          >> tot: %.2f\n' % (tot_err)
        print(eval_summary)
        
        acc_error = acc_error_h36m / n
        eval_summary = 'Protocol ' + str(self.protocol) + ' H36M ACCEL (mm/s^2) >> tot: %.2f\n ' % (acc_error)
        print(eval_summary)