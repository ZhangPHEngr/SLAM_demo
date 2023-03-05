# -*- coding: utf-8 -*-
# @Time    : 3/5/23 6:17 PM
# @Author  : ZhangP.H
# @File    : p3p.py
# @Software: PyCharm
import cv2
import numpy as np

np.random.seed(42)


def p3p(image_points, world_points, camera_matrix, distortion_coeffs=np.zeros((5, 1))):
    """
    :param image_points: 物体上三个点在图像上的像素坐标，Nx2的数组
    :param world_points: 物体上三个点的真实坐标，Nx3的数组
    :param camera_matrix: 相机内参矩阵，3x3的数组
    :param distortion_coeffs: 畸变系数，5x1的数组
    :return: rvecs, tvecs
    """
    # 将图像点转化为归一化平面坐标系下的坐标
    image_points_norm = cv2.undistortPoints(np.expand_dims(image_points, axis=1), camera_matrix, distortion_coeffs)
    image_points_norm = image_points_norm.squeeze()

    # 计算相机坐标系下物体上三个点的坐标
    world_points = np.hstack((world_points, np.ones((len(world_points), 1), dtype=np.float32)))
    world_points_camera = np.dot(world_points, cv2.invert(camera_matrix)[1].T)

    # 依次计算每组解
    solutions = []
    for i in range(len(world_points)):
        p1 = world_points_camera[i]
        p2 = world_points_camera[(i + 1) % 3]
        p3 = world_points_camera[(i + 2) % 3]
        u1 = image_points_norm[i]
        u2 = image_points_norm[(i + 1) % 3]
        u3 = image_points_norm[(i + 2) % 3]

        # 构建系数矩阵
        A = np.zeros((3, 4), dtype=np.float32)
        A[:3, :3] = np.vstack((p1, p2, p3)).T
        A[:, 3] = -np.array([u1[0], u2[0], u3[0]])

        # 计算解
        _, _, vt = cv2.SVDecomp(A)
        V = vt.T
        X = V[:, -1]

        # 将解转化为相机坐标系下的旋转向量和平移向量
        rvec = X[:3]
        norm_r = np.linalg.norm(rvec)
        rvec = rvec / norm_r if norm_r != 0 else np.zeros(3)
        tvec = X[3:] / norm_r if norm_r != 0 else np.zeros(3)

        # 将解添加到解集中
        solutions.append((rvec, tvec))

    # 从解集中选择最佳解
    best_rvec, best_tvec, num_inliers = cv2.solvePnPRansac(world_points, image_points, camera_matrix, distortion_coeffs, reprojectionError=5.0)

    # 计算每个解的重投影误差，选择重投影误差最小的解作为最终解
    # min_error = float('inf')
    # for rvec, tvec in solutions:
    #     _, image_points_proj = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, distortion_coeffs)
    #     error = np.mean(np.linalg.norm(image_points - image_points_proj.squeeze(), axis=1))
    #     if error


"""
SOLVEPNP_ITERATIVE Iterative method is based on Levenberg-Marquardt optimization. In this case, the function finds such a pose that minimizes reprojection error, that is the sum of squared distances between the observed projections imagePoints and the projected (using projectPoints() ) objectPoints.
SOLVEPNP_P3P Method is based on the paper of X.S. Gao, X.-R. Hou, J. Tang, H.-F. Chang “Complete Solution Classification for the Perspective-Three-Point Problem”. In this case, the function requires exactly four object and image points.
SOLVEPNP_EPNP Method has been introduced by F.Moreno-Noguer, V.Lepetit and P.Fua in the paper “EPnP: Efficient Perspective-n-Point Camera Pose Estimation”.
"""
if __name__ == '__main__':
    # Made up projective matrix
    K = np.array([[160, 0, 320], [0, 120, 240], [0, 0, 1]]).astype(dtype=np.float32)
    dist_coeffs = np.zeros((4, 1)).astype(dtype=np.float32)  # Assuming no lens distortion - the image points were obtained from a rectified image

    # A pose
    R_gt = np.array(
        [
            [-0.48048015, 0.1391384, -0.86589799],
            [-0.0333282, -0.98951829, -0.14050899],
            [-0.8763721, -0.03865296, 0.48008113],
        ]
    )
    t_gt = np.array([-0.10266772, 0.25450789, 1.70391109])

    # A B C in world cord
    pts = 0.6 * (np.random.random((4, 3)) - 0.5).astype(dtype=np.float32)
    # p in camera cord
    pts_2d = (pts @ R_gt.T + t_gt) @ K.T
    pts_2d = (pts_2d / pts_2d[:, -1, None])[:, :-1].astype(dtype=np.float32)

    # Compute pose candidates. the problem is not minimal so only one
    # will be provided
    # poses = p3p(pts_2d, pts, K, )
    #
    # # Print results
    # print("R (ground truth):", R_gt, sep="\n")
    # print("t (ground truth):", t_gt)
    #
    # print("Nr of possible poses:", len(poses))
    # for i, pose in enumerate(poses):
    #     R, t = pose
    #
    #     # Project points to 2D
    #     pts_2d_est = (pts @ R.T + t) @ K.T
    #     pts_2d_est = (pts_2d_est / pts_2d_est[:, -1, None])[:, :-1]
    #     err = np.mean(np.linalg.norm(pts_2d - pts_2d_est, axis=1))
    #
    #     print("Estimate -", i + 1)
    #     print("R (estimate):", R, sep="\n")
    #     print("t (estimate):", t)
    #     print("Mean error (pixels):", err)

    (success, rotation_vector, translation_vector) = cv2.solvePnP(pts, pts_2d, K, dist_coeffs, flags=cv2.SOLVEPNP_P3P)

    # converting the rotation vector to a rotation matrix
    rot_mat, jacobian = cv2.Rodrigues(rotation_vector)

    # printing all the parameters
    print("Camera Matrix :\n {0}".format(K))
    print("Rotation Vector:\n {0}".format(rotation_vector))
    print("Translation Vector:\n {0}".format(translation_vector))
    print("Rotation Matrix: \n {0}".format(rot_mat))
    print("saved world_points, rotation matrix and translation matrix text files \n ")
