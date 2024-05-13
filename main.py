########################################################################
#
# Copyright (c) 2024, ETS.
#
# TIME 4-29-2024
# THIS SOFTWARE IS PROVIDED BY LIUYANJUN. ALL IMPLEMENTATIONS INCLUDE 
# USE YOLOV5 SEGMENT 2D RGB IMAGE TO RETRIVE THE SEGMENTATION OF WOOD.
# AND THEN, THE 2D SEGMENTATION GUIDES TO SEGMENT 3D POINTCLOUD FOR WOOD POINTCLOUD.
# 
# NEXT, CALCULATE THE MAJOR AND MINOR DIAMETER OF WOOD END FACE THROUGH ELLIPSE FITTING 
# METHOD OR CUT_OR_FILL METHOD. 
#
########################################################################




import open3d as o3d
import numpy as np
import tqdm
import copy
import sys
import torch
import cv2 as cv
from PIL import Image
from pathlib import Path
from torchvision import transforms
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from collections import Counter

from segmentor.models.common import DetectMultiBackend
from segmentor.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from segmentor.utils.plots import Annotator, colors, save_one_box
from segmentor.utils.segment.general import masks2segments, process_mask
from segmentor.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from segmentor.utils.torch_utils import select_device, smart_inference_mode


class point_cloud():
    def __init__(self, 
                 # for yolov5
                 source="./segmentor/data/images",
                 weights="./segmentor/checkpoints/best.pt",
                 data="./segmentor/data/coco128.yaml",
                 project="./segmentor/runs/predict-seg",
                 imgsz=(1280, 1280),
                 dnn=False,
                 half=False,
                 augment=False,

                 conf_thres=0.25,
                 iou_thres=0.45,
                 max_det=1000,
                 classes=None,
                 angostic_nms=False,
                 vid_stride=1, 

                 save_crop=False,
                 device='0',
                 visualize=False
                 # for 3D point cloud
                ):
        
        self.source = source
        self.weights = weights
        self.data = data
        self.project = project
        self.imgsz = imgsz
        self.visualize = visualize
        
        self.dnn = dnn
        self.half = half
        self.augment = augment
        
        self.conf_thres = conf_thres
        self.max_det = max_det
        self.classes = classes
        self.angostic_nms = angostic_nms

        self.save_crop = save_crop
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.bs = 1
        self.iou_thres = iou_thres
        self.vid_stride = vid_stride

        self.path = "./results"
        self.source = "./images"

    def _compute_camera_extri(self, img_l, img_r, K):
        sift = cv2.xfeatures2d.SIFT_create()
        
        kp_l, des_l = sift.detectAndCompute(img_l, None)
        kp_r, des_r = sift.detectAndCompute(img_r, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_l, des_r, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        essential_mat, mask = cv2.findEssentialMat(kp_l, kp_r, K)
        _, R, T, _ = cv2.recoverPose(essential_mat, kp_l, kp_r, K)      
        
        return R, T
    
    def _get_score(self, input_image=None):
        """
        Get the segmentation score of image.

        :param input_image:

        :return segmentation score: a tensor H*W*(wood_num+1), class 0 is the background, other classes are the different woods.
        """
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        # Dataloader
        dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)

        # vid_path, vid_writer = [None] * self.bs, [None] * self.bs
        output_reassign_softmax = []

        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                # if im.shape[0] != 3:
                #     im = im.permute(2, 0, 1)
                #     print("DONE")
                im = im.half() if self.model.fp16 else im.float()
                im /= 255
                if len(im.shape) == 3:
                    im = im[None]

            with dt[1]:
                self.visualize = increment_path(self.save_dir / Path(self.path).stem, mkdir=True) if self.visualize else False
                pred, proto = self.model(im, self.augment, visualize=self.visualize)[:2]
            
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.angostic_nms, max_det=self.max_det, nm=32)
            
            outputs_reassign = []
            for i, det in enumerate(pred):
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = self.path

                p = Path(p) # to path
                # save_path = str(self.save_dir / p.name)
                # txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}') # im.txt
                s += '%gx%g ' % im.shape[2:]
                # imc = im0.copy() if self.save_crop else im0

                annotator = Annotator(im0, line_width=3, example=str('wood'))
                if len(det):
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # CHW
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                    segments = reversed(masks2segments(masks))
                    segments = [scale_segments(im.shape[2:], x, im0.shape, normalize=False) for x in segments]
                    # segments = np.array(segments, dtype=np.uint8)
                    origin_masks = []
                    for i in range(len(segments)):
                        segment = np.array(segments[i], dtype=np.int32)
                        # segment = segment.reshape((-1, 1, 2))
                        mask = np.zeros((im0.shape[0], im0.shape[1], 3), dtype=np.uint8)
                        cv2.fillPoly(mask, [segment], color=(255,255,255))
                        
                        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        binary_mask01 = mask_gray / 255
                        # cv2.imwrite('./Mask.png', binary_image)
                        origin_masks.append(binary_mask01)

                    origin_masks = torch.tensor(origin_masks)   # (C, H ,W)
                    # Mask plotting
                    retina_masks = False
                    # annotator.masks(masks,
                    #                 colors=[colors(x, True) for x in det[:, 5]],
                    #                 im_gpu=None if retina_masks else im[i])

                    output_permute = origin_masks.permute(1, 2, 0) # HWC
                    # output_probability, output_predictions = output_permute.max(2)
                    woods_masks = output_permute.sum(dim=2)
                    wood_num = origin_masks.shape[0]

                    background_mask = ~(woods_masks == 1)
                    # cv2.imwrite("./wood.png", np.array(woods_masks * 255).astype(np.uint8))
                    # cv2.imwrite("./back.png", np.array(background_mask*255, dtype=np.uint8))
                    softmax = torch.nn.Softmax(dim=2)

                    output_reassign = torch.zeros(output_permute.size(0), output_permute.size(1), wood_num + 1)
                    output_reassign[:, :, 0] = background_mask
                    # cv2.imwrite("./back.png", np.array(output_reassign[:,:,0]*255, dtype=np.uint8))
                    # each wood mask
                    for i in range(wood_num):
                        output_reassign[:, :, i+1] = output_permute[:, :, i]
                    # output_reassign_softmax.append(output_reassign.cpu().numpy())
                    output_reassign = output_reassign.cpu().numpy()
                outputs_reassign.append(output_reassign)
                
                # im0 = annotator.result()
                # cv2.imwrite("./res.png", im0)
        
        return outputs_reassign

    
    def _cam_to_point(self, pointcloud, projection_mats):
        """
        Takes in points in world coords, return points in camera coords

        :param pointcloud: (n_points, 3) np.array (x,y,z) in world coordinates

        :return point_cam_coords: (n_points, 3) np.array (x,y,z) in camera coordinates
        """
        point_world_coords = copy.deepcopy(pointcloud)
        point_world_coords = np.hstack((point_world_coords, np.ones((point_world_coords.shape[0], 1))))    # for multiplying with homogeneous matrix
        point_cam_coords = point_world_coords
        
        return point_cam_coords
    
    def _cal_rotation_mat(self, normal_vector):
        """
        Calculate the rotation matrix to rotate the fitted plane parallel to xy plane.
        """
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        angle = np.arccos(normal_vector[2])
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                    [np.sin(angle), np.cos(angle), 0],
                                    [0, 0, 1]])
        
        return rotation_matrix
    
    def _align_segmentation_pointcloud(self, segmentation_l, segmentation_r, calib, points):
        """
        Projects points onto segmentation map, appends class score each point projects onto.

        :params segmenttaion_l: (H, W, background+wood_num), the wood_num is number of the wood for our work, segmentation score of left camera.
        :params segmentation_r: (H, W, background+wood_num), segmentation score of right camera.
        :params calib: ("intri", "extri"), dict{numpy.array}, camera calibration parameters.
        :params points: (n, 3), numpy.aray, input pointcloud coordinations.

        :return: (n, 3+1+wood_num), "3" is the xyz coordinations, "1" is the background mask, "wood_num" is the number of woods. The painted points.
        """
        # flip
        segmentation_l = cv.flip(segmentation_l, 1)
        point_coords = points
        projection_mats = calib
        point_cam_coords = self._cam_to_point(points, projection_mats)

        # right
        # point_cam_coords[:, -1] = 1 #homogenous coords for projection
        # # project points in poincloud coordinate system to captured images by camera, (x,y,z)->(u,v,w)->(u/w,v/w,1) 
        # points_projected_on_mask_r = calib['intri_r'].dot(point_coords.transpose())
        # points_projected_on_mask_r = points_projected_on_mask_r.transpose()
        # points_projected_on_mask_r = points_projected_on_mask_r / (points_projected_on_mask_r[:,2].reshape(-1,1))

        # # the coordination of points that projected to image plane must in the range (0, w)
        # true_where_x_on_img_r = (0 < points_projected_on_mask_r[:, 0]) & (points_projected_on_mask_r[:, 0] < segmentation_r.shape[1])
        # # the coordination of points that projected to image plane must in the range(0, h)
        # true_where_y_on_img_r = (0 < points_projected_on_mask_r[:, 1]) & (points_projected_on_mask_r[:, 1] < segmentation_r.shape[0])
        # true_where_point_on_img_r = true_where_x_on_img_r & true_where_y_on_img_r   # bool

        # points_projected_on_mask_r = points_projected_on_mask_r[true_where_y_on_img_r]
        # points_projected_on_mask_r = np.floor(points_projected_on_mask_r).astype(int)
        # points_projected_on_mask_r = points_projected_on_mask_r[:, :2]  # (u, v)

        # left camera
        points_projected_on_mask_l = calib['intri_l'].dot(calib['trans'].dot(point_cam_coords.transpose()))
        points_projected_on_mask_l = points_projected_on_mask_l.transpose()
        points_projected_on_mask_l = points_projected_on_mask_l/(points_projected_on_mask_l[:,2].reshape(-1,1))

        true_where_x_on_img_l = (0 < points_projected_on_mask_l[:, 0]) & (points_projected_on_mask_l[:, 0] < segmentation_l.shape[1]) #x in img coords is cols of img
        true_where_y_on_img_l = (0 < points_projected_on_mask_l[:, 1]) & (points_projected_on_mask_l[:, 1] < segmentation_l.shape[0])
        true_where_point_on_img_l = true_where_x_on_img_l & true_where_y_on_img_l

        points_projected_on_mask_l = points_projected_on_mask_l[true_where_point_on_img_l] # filter out points that don't project to image
        points_projected_on_mask_l = np.floor(points_projected_on_mask_l).astype(int) # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask_l = points_projected_on_mask_l[:, :2] # drops homogenous coord 1 from every point, giving (N_pts, 2) int array
        wood_points = points
        # true_where_point_on_both_img = true_where_point_on_img_l & true_where_point_on_img_r
        # true_where_point_on_img = true_where_point_on_img_l | true_where_point_on_img_r
        true_where_point_on_img = true_where_point_on_img_l    # for one img

        # row -> y, col -> x
        # scores (h, w, c)-> points_scores (h*w, c)
        # point_scores_r = segmentation_r[points_projected_on_mask_r[:, 1], points_projected_on_mask_r[:, 0]].reshape(-1, segmentation_r.shape[2])
        point_scores_l = segmentation_l[points_projected_on_mask_l[:, 1], points_projected_on_mask_l[:, 0]].reshape(-1, segmentation_l.shape[2])

        # augmented_pointcloud includes position and segmentation score.
        augmented_points = np.concatenate((points, np.zeros((points.shape[0], segmentation_l.shape[2]))), axis=1)
        # augmented_lidar[true_where_point_on_img_r, -segmentation_r.shape[2]:] += point_scores_r
        augmented_points[true_where_point_on_img_l, -segmentation_l.shape[2]:] += point_scores_l
        # augmented_lidar[true_where_point_on_both_img, -segmentation_r.shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_img, -segmentation_r.shape[2]:]
        augmented_points = augmented_points[true_where_point_on_img]  # (x,y,z, 0class, 1class, 2class, ...)

        return augmented_points
    
    def _fit_plane_LSM(self, pcd):
        points = np.asarray(pcd.points)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        N = len(points)
        A = np.array([[sum(x ** 2), sum(x * y), sum(x)],
                      [sum(x * y), sum(y ** 2), sum(y)],
                      [sum(x), sum(y), N]])
        B = np.array([[sum(x * z), sum(y * z), sum(z)]])
        X = np.linalg.solve(A, B.T)
        a, b, c, d = X[0][0], X[1][0], -1, X[2][0]

        return a, b, c, d


    def cal_diameter(self, points, calib, method="cut_or_fill", SegmentPlane=False, vis=True):   # "ellipse, cut_or_fill"
        """
        Fit ellipse in 2D space.

        :param points: ply pointclodu, indicates the input wood points.

        :return: infos of diameter.
        """
        if SegmentPlane:
            # find plane use RANSAC
            distance_thres = 0.01
            ransac_n = 10
            num_iterations = 1000
            plane_model, inliers = points.segment_plane(distance_thres, ransac_n, num_iterations)
            wood_face = points.select_by_index(inliers)

            # points_coord = np.array(points.points)
            # dbscan = DBSCAN(eps=0.004, min_samples=20)
            # dbscan.fit(points_coord)
            # labels = dbscan.labels_
            # unique_labels, counts = np.unique(labels, return_counts=True) 
            # largest_cluster_label = unique_labels[np.argmax(counts)]
            # wood_face_points = points_coord[labels == largest_cluster_label]
            # wood_face = o3d.geometry.PointCloud()
            # wood_face.points = o3d.utility.Vector3dVector(wood_face_points)

            [a, b, c, d] = plane_model
            # print(f"plane equations: {a:.2f}x + {b:.2f}y + {c:.2f} + {d:.2f} = 0")
        else:
            # find plane use DBSCAN (clusting)
            points_coord = np.array(points.points)
            dbscan = DBSCAN(eps=0.004, min_samples=20)
            dbscan.fit(points_coord)
            labels = dbscan.labels_

            # count point number of each class
            unique_labels, counts = np.unique(labels, return_counts=True) 
            largest_cluster_label = unique_labels[np.argmax(counts)]
            wood_face_points = points_coord[labels == largest_cluster_label]
            wood_face = o3d.geometry.PointCloud()
            wood_face.points = o3d.utility.Vector3dVector(wood_face_points)
            # fit plane
            a, b, c, d = self._fit_plane_LSM(wood_face)
        
        if vis:
            o3d.visualization.draw_geometries([wood_face])
        plane_normal = [a, b, c]
        
        # projection points to fitted plane
        plane_seeds = []
        for xyz in wood_face.points:
            x, y, z = xyz
            t = -(a * x + b * y + c * z + d) / (a * a + b * b + c * c)
            xi = a*t + x
            yi = b*t + y
            zi = c*t + z
            plane_seeds.append((xi, yi, zi))
        

        rotation_matrix = self._cal_rotation_mat(plane_normal)    # for RANSCAN

        # theta = np.arctan(-b/a)
        # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], 
        #                                 [np.sin(theta), np.cos(theta), 0], 
        #                                 [0, 0, 1]])

        # rotate point
        if vis:
            plane_cloud = o3d.geometry.PointCloud()
            plane_cloud.points = o3d.utility.Vector3dVector(plane_seeds)
            o3d.visualization.draw_geometries([plane_cloud])

        rotated_point = np.dot(plane_seeds, rotation_matrix.T)
        xy_plane_cloud = o3d.geometry.PointCloud()
        xy_plane_cloud.points = o3d.utility.Vector3dVector(rotated_point)
        
        # reduce dimension
        imgsz = 2080
        offset = 1
        # z_value = rotated_point[:, 2][0]
        counter = Counter(rotated_point[:, 2])
        z_value, count = counter.most_common(1)[0]
        # z_value = -(a*x + b*y + d) / c
        reduced_points = (rotated_point[:, :2] + offset) * (imgsz / 2)
        xy_points = np.array(reduced_points, dtype=np.float32)
        
        draw_plane = np.ones((imgsz, imgsz, 3), dtype=np.uint8) * 0
        draw_plane = cv.cvtColor(draw_plane, cv.COLOR_BGR2GRAY)
        draw_point = tuple(map(tuple, reduced_points.astype(int)))

        for point in draw_point:
            cv.circle(draw_plane, point, 0, 255, 3)

        # connected components process
        draw_plane = self._connectedComponents_process(draw_plane)

        if method == "ellipse":
            data_container = self._use_ellipse(draw_plane, imgsz)
        elif method == "cut_or_fill":
            data_container = self._use_circle_cutOrFill(draw_plane, imgsz, vis=vis)
        
        """ major diameter endpoints project back to origin image pixel coordination. 
        
        data_container = {"center": center, 
                          "diameter": (major_diameter, minor_diameter), 
                          "start_end_coord_major": max_diameter_line, 
                          "start_end_coord_minor": min_diameter_line}
        """
        # translate
        data_container["center"] -= offset
        data_container["start_end_coord_major"] -= offset 
        data_container["start_end_coord_minor"] -= offset

        # append value of z axis
        # modify center coord
        center_array = data_container["center"]
        center_array = np.expand_dims(center_array, axis=0)
        center_array = np.append(center_array, z_value)

        # modify major axis coord
        max_diameter_startpoint = data_container["start_end_coord_major"][0]
        max_diameter_endpoint = data_container["start_end_coord_major"][1]

        max_diameter_startpoint = np.expand_dims(max_diameter_startpoint, axis=0)
        max_diameter_startpoint = np.append(max_diameter_startpoint, z_value)
        max_diameter_startpoint = np.expand_dims(max_diameter_startpoint, axis=0)
        max_diameter_endpoint = np.expand_dims(max_diameter_endpoint, axis=0)
        max_diameter_endpoint = np.append(max_diameter_endpoint, z_value)
        max_diameter_endpoint = np.expand_dims(max_diameter_endpoint, axis=0)
        max_diameter_array = np.concatenate([max_diameter_startpoint, max_diameter_endpoint], axis=0)
        
        # modify minor axis coord
        min_diameter_startpoint = data_container["start_end_coord_minor"][0]
        min_diameter_endpoint = data_container["start_end_coord_minor"][1]

        min_diameter_startpoint = np.expand_dims(min_diameter_startpoint, axis=0)
        min_diameter_startpoint = np.append(min_diameter_startpoint, z_value)
        min_diameter_startpoint = np.expand_dims(min_diameter_startpoint, axis=0)
        min_diameter_endpoint = np.expand_dims(min_diameter_endpoint, axis=0)
        min_diameter_endpoint = np.append(min_diameter_endpoint, z_value)
        min_diameter_endpoint = np.expand_dims(min_diameter_endpoint, axis=0)
        min_diameter_array = np.concatenate([min_diameter_startpoint, min_diameter_endpoint], axis=0)
        
        # rotate points back to origin plane
        inverse_rotate_mat = np.linalg.inv(rotation_matrix)
        center_array = np.dot(inverse_rotate_mat, center_array.transpose()).transpose()
        max_diameter_array = np.dot(inverse_rotate_mat, max_diameter_array.transpose()).transpose()
        min_diameter_array = np.dot(inverse_rotate_mat, min_diameter_array.transpose()).transpose()

        # project points to image plane
        homo_center_array = np.append(center_array, 1)
        homo = np.ones((max_diameter_array.shape[0], 1))
        homo_max_diameter_array = np.concatenate((max_diameter_array, homo), axis=1)
        homo_min_diameter_array = np.concatenate((min_diameter_array, homo), axis=1)
        center_array = calib['intri_l'].dot(calib['trans'].dot(homo_center_array.transpose()))
        center_array = (center_array / center_array[2])[:2].astype(int)
        max_diameter_array = calib['intri_l'].dot(calib['trans'].dot(homo_max_diameter_array.transpose())).transpose()
        max_diameter_array = (max_diameter_array / max_diameter_array[:, 2].reshape(-1, 1))[:, :2].astype(int)
        min_diameter_array = calib['intri_l'].dot(calib['trans'].dot(homo_min_diameter_array.transpose())).transpose()
        min_diameter_array = (min_diameter_array / min_diameter_array[:, 2].reshape(-1, 1))[:, :2].astype(int)

        data_container["center"] = center_array
        data_container["start_end_coord_major"] = max_diameter_array
        data_container["start_end_coord_minor"] = min_diameter_array

        return data_container

    def _opening_process(self, img):
        kernel = np.ones((1, 1), np.uint8)
        opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

        return opening
    
    def _connectedComponents_process(self, img):
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=8)

        min_area = 4
        filtered_image = np.zeros_like(img)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered_image[labels == i] = 255
        
        return filtered_image
    
    def _use_circle_cutOrFill(self, draw_plane, imgsz, method="fast_opposite", CutAndFill=False, vis=True): # "fast_opposite, cross_center"
        contour, _ = cv.findContours(draw_plane, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        origin_plane = draw_plane.copy()
        # find the equivalent radius of wood contour
        contr = max(contour, key=len)
        # for contr in contour:
        moment = cv.moments(contr)
        center_x = int(moment['m10'] / moment['m00'])
        center_y = int(moment['m01'] / moment['m00'])
        center = (center_x, center_y)

        area = cv.contourArea(contr)
        equivalent_radius = int(np.sqrt(area / np.pi))

        if CutAndFill:
            cv.circle(draw_plane, center, equivalent_radius, 0, -1)
            
            """ cut or keep external field """
            ex_contours, _ =  cv.findContours(draw_plane, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # cv.drawContours(draw_plane, ex_contours, -1, 120, 1)
            ex_contour = max(ex_contours, key=len)
            # for ex_contour in ex_contours:
            max_distance = 0
            for point in ex_contour:
                distance = np.sqrt((point[0][0] - center_x)**2 + (point[0][1] - center_y)**2)
                if distance > max_distance:
                    max_distance = distance
            if max_distance > equivalent_radius * 1.1:
                cv.fillPoly(origin_plane, [ex_contour], 0) # modify in the origin image


            """ fill """
            new_contours, _ = cv.findContours(origin_plane, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            wood_contour = max(new_contours, key=len)
            # calculate each contour
            min_distance = equivalent_radius
            depression = []
            i = 0
            while i < wood_contour.shape[0]:
                # calculate distance between each point and center.
                deep_index = 0
                min_distance = equivalent_radius
                point = wood_contour[i]
                distance = np.sqrt((point[0][0] - center_x)**2 + (point[0][1] - center_y)**2)
                tag = i
                is_start = False    # mark whether to enter a depression.
                while distance < equivalent_radius:
                    # save the start point index
                    if i == tag:
                        is_start = True
                        start = tag
                    if distance < min_distance:
                        min_distance = distance
                        deep_index = i
                        depression_info_dict = {"index": deep_index, "dist":min_distance, "start": start}
                    i += 1
                    if i < wood_contour.shape[0]:
                        distance = np.sqrt((wood_contour[i][0][0] - center_x)**2 + (wood_contour[i][0][1] - center_y)**2)
                    else:
                        break
                
                # add the end point index
                if is_start:
                    depression_info_dict["end"] = i - 1
                if deep_index != 0 and i != 0:
                    depression.append(depression_info_dict)
                i += 1
                
            # fill these depression with rule
            for i in range(len(depression)):
                dist = depression[i]["dist"]
                if 0.8 * equivalent_radius > dist:  # 1/2 depth
                    new_radius = 1/2 * (equivalent_radius + dist)
                    depression[i]["dist"] = new_radius
                else:
                    depression[i]["dist"] = equivalent_radius
                
                # calculate the angle of the fan 
                start_x = wood_contour[depression[i]["start"]][0][0]
                start_y = wood_contour[depression[i]["start"]][0][1]
                end_x = wood_contour[depression[i]["end"]][0][0]
                end_y = wood_contour[depression[i]["end"]][0][1]

                theta = np.arctan2(start_y - center[1], start_x - center[0])
                phi = np.arctan2(end_y - center[1], end_x - center[0])

                # draw fan
                radius = int(depression[i]["dist"])
                cv.ellipse(origin_plane, center, axes=(radius, radius), angle=0, startAngle=theta, endAngle=phi, color=255, thickness=-1)


        # calculate major diameter and minor diameter
        final_contours, _ = cv.findContours(origin_plane, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        final_wood_contour = max(final_contours, key=len)

        # fill the whole area with new contour
        # cv.fillPoly(origin_plane, [final_wood_contour], 255)

        if method == "fast_opposite":
            max_diameter_line, min_diameter_line, max_dist, min_dist = self._fast_opposite_measure(final_wood_contour, center, equivalent_radius)
        elif method == "clock_rotate":
            max_diameter_line, min_diameter_line, max_dist, min_dist = self._clock_rotate_measure(final_wood_contour, center, equivalent_radius)
        # else:
        #     max_diameter_line, min_diameter_line, max_dist, min_dist = self._cross_center_measure(final_wood_contour, center, equivalent_radius, origin_plane)

        major_diameter = max_dist / (imgsz / 2)
        minor_diameter = min_dist / (imgsz / 2)
        print(f"major length: {major_diameter}, minor length: {minor_diameter}")
        if vis:
            cv.line(origin_plane, max_diameter_line[0], max_diameter_line[1], 100, 1)   # major line
            cv.line(origin_plane, min_diameter_line[0], min_diameter_line[1], 200, 1)   # minor line

            cv.imshow("test", origin_plane)
            cv.waitKey(0)
            cv.destroyAllWindows()

        center = np.array(center) / (imgsz / 2)
        max_diameter_line = max_diameter_line / (imgsz / 2)
        min_diameter_line = min_diameter_line / (imgsz / 2)
        data_container = {"center": center, 
                          "diameter": (major_diameter, minor_diameter), 
                          "start_end_coord_major": max_diameter_line, 
                          "start_end_coord_minor": min_diameter_line}

        return data_container

    def _clock_rotate_measure(self, final_wood_contour, center, equivalent_radius):

        raise NotImplementedError
    
    def _cross_center_measure(self, final_wood_contour, center, equivalent_radius, img):
        # hull
        hull = cv.convexHull(final_wood_contour)
        cv.polylines(img, [hull], True, 200, 2)
        cv.imshow("test", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        raise NotImplementedError

    def _fast_opposite_measure(self, final_wood_contour, center, equivalent_radius):
        max_dist = 0
        min_dist = equivalent_radius * 3
        
        max_diameter_line = []
        min_diameter_line = []

        num_points = len(final_wood_contour)
        offset = num_points * 0.005
        scan_area = (int(num_points / 2 - offset), int(num_points / 2 + offset))
        
        for i in range(num_points):
            for j in range((i+scan_area[0]) % num_points, (i+scan_area[1]) % num_points):
                start_index = i
                end_index = j
                distance = np.sqrt((final_wood_contour[start_index][0][0] - final_wood_contour[end_index][0][0])**2 + (final_wood_contour[start_index][0][1] - final_wood_contour[end_index][0][1])**2)
                if distance > max_dist:
                    max_dist = distance
                    # start point, end point
                    max_diameter_line = np.array(((final_wood_contour[start_index][0][0], final_wood_contour[start_index][0][1]), (final_wood_contour[end_index][0][0], final_wood_contour[end_index][0][1])))
                if distance < min_dist:
                    min_dist = distance
                    min_diameter_line = np.array(((final_wood_contour[start_index][0][0], final_wood_contour[start_index][0][1]), (final_wood_contour[end_index][0][0], final_wood_contour[end_index][0][1])))

        return max_diameter_line, min_diameter_line, max_dist, min_dist


    def _use_ellipse(self, draw_plane, imgsz):
        contour, _ = cv.findContours(draw_plane, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(draw_plane, contour, -1, 125, 1)
        
        for cnt in contour:
            external_ellipse = cv.fitEllipse(cnt)
            # cv.ellipse(draw_plane, external_ellipse, 200, 3)
            center, axes, angle = external_ellipse
            center = center / (imgsz / 2)
            major_diameter = max(axes) / (imgsz / 2)
            minor_diameter = min(axes) / (imgsz / 2)
            axes = axes / (imgsz / 2)
            print("Results of fitting ellipse as follow:")
            print(f"center:{center}, major axis length:{major_diameter}, minor axis length:{minor_diameter}.")
            # long endpoint
            major_end0 = (int(center[0] + 0.5 * max(axes) * np.cos(np.radians(angle))), 
                        int(center[1] + 0.5 * max(axes) * np.sin(np.radians(angle))))
            major_end1 = (int(center[0] - 0.5 * max(axes) * np.cos(np.radians(angle))), 
                        int(center[1] - 0.5 * max(axes) * np.sin(np.radians(angle))))
            max_diameter_line = np.array((major_end0, major_end1))
            
            # short endpoint
            minor_end0 = (int(center[0] + 0.5 * min(axes) * np.cos(np.radians(angle + 90))), 
                        int(center[1] + 0.5 * min(axes) * np.sin(np.radians(angle + 90))))
            minor_end1 = (int(center[0] - 0.5 * min(axes) * np.cos(np.radians(angle + 90))), 
                        int(center[1] - 0.5 * min(axes) * np.sin(np.radians(angle + 90))))
            min_diameter_line = np.array((minor_end0, minor_end1))

            cv.line(draw_plane, major_end0, major_end1, 100, 1)
            cv.line(draw_plane, minor_end0, minor_end1, 200, 1)

        cv.imshow("test", draw_plane)
        cv.waitKey(0)
        cv.destroyAllWindows()

        data_container = {"center": center, 
                          "diameter": (major_diameter, minor_diameter), 
                          "start_end_coord_major": max_diameter_line, 
                          "start_end_coord_minor": min_diameter_line}

        return data_container

    def match_endface(self, points):
        raise NotImplementedError
    

    def run(self, img_l=None, img_r=None, points=None):
        point_cloud = o3d.io.read_point_cloud("./pointcloud/Pointcloud_NEURAL.ply")
        points = np.asarray(point_cloud.points)
        img_l = cv2.imread("./images/l_img.png")
        # img_r = cv2.imread("./images/r_img.png")

        # scores_from_cam_l = self._get_score(img_l)
        # scores_from_cam_r = self._get_score(img_r)
        scores_lr = self._get_score()
        scores_l = scores_lr[0]
        # scores_r = scores_lr[1]
        scores_r = None
        
        # calib = self.get_calib()
        calib = {"intri_r":np.array([[1.05742368e3, 0.0000000000, 1.12879931e3, 0],
                                    [0.00000000000, 1.05756302e3, 6.24335381e2, 0],
                                    [0.00000000000, 0.0000000000, 1.00000000e0, 0]]), 
                "extri_r":[], 
                "intri_l":np.array([[1.04925086e3, 0.0000000000, 1.12862928e3, 0],
                                    [0.0000000000, 1.05047247e3, 6.18653411e2, 0],
                                    [0.0000000000, 0.0000000000, 1.0000000000, 0]]), 
                "extri_l":[],
                "trans": np.array([[1.0, 0.0, 0.0, 0.11981524658/2],#0.11981524658#0.480
                                   [0.0, 1.0, 0.0,   0.0      ],    # 0.2
                                   [0.0, 0.0, 1.0,   0.0       ],
                                   [0.0, 0.0, 0.0,   1.0       ]])
                }
                            
        dist = {"dist_l": np.array([[0.01448826, -0.03668633, -0.0006209, 0.00028326, 0.02701385]]), 
                "dist_r": np.array([[1.36497253e-2, -4.19849978e-2, 2.78190673e-5, -7.10696255e-4, 3.01986427e-2]])}

        points = self._align_segmentation_pointcloud(scores_l, scores_r, calib, points)
        

        img_l = cv.flip(img_l, 1)
        for i in range(4, points.shape[1]):
            wood_points = points[points[:, i] == 1]
            seg_pointCloud = o3d.geometry.PointCloud()
            seg_pointCloud.points = o3d.utility.Vector3dVector(wood_points[:, 0:3])
            o3d.io.write_point_cloud("./point_cloud.ply", seg_pointCloud)

            # wood_points = points[:, :, 0]

            try:
                data_container = self.cal_diameter(seg_pointCloud, calib)
            except:
                data_container = []

            cv.circle(img_l, center=data_container["center"], radius=1, color=(0, 255, 0), thickness=1)
            cv.line(img_l, data_container["start_end_coord_major"][0], data_container["start_end_coord_major"][1], color=(255, 0, 0), thickness=1)
            cv.line(img_l, data_container["start_end_coord_minor"][0], data_container["start_end_coord_minor"][1], color=(0, 0, 255), thickness=1)
            # self.cal_center(wood_points)
            # self.match_endface(wood_points)
            # self.cal_diameter(wood_points)
        img_l = cv.flip(img_l, 1)
        cv.imwrite(f"./origin_latest.png", img_l)
        

    def accept_data(self, seg_score, img, depth):
        """
        accept data from 2D processing stage.
        
        :params seg_score: (H, W, wood_num), the segmentation score from the camera.
        :params img: (H, W, C), RGB images from camera.
        :params depth: (H, W, 1), depth map from camera.
        """
        raise NotImplementedError

        


if __name__=="__main__":
    data = []
    model = point_cloud()
    model.run()
