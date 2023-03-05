# -*- coding: utf-8 -*-
# @Time    : 3/5/23 6:17 PM
# @Author  : ZhangP.H
# @File    : p3p.py
# @Software: PyCharm
import cv2
import numpy as np

np.random.seed(42)

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

    # ------------------------------------------solve---------------------------------------------------------- #
    (success, rotation_vector, translation_vector) = cv2.solvePnP(pts, pts_2d, K, dist_coeffs, flags=cv2.SOLVEPNP_P3P)

    # converting the rotation vector to a rotation matrix
    rot_mat, jacobian = cv2.Rodrigues(rotation_vector)

    # printing all the parameters
    print("Camera Matrix :\n {0}".format(K))
    print("Rotation Vector:\n {0}".format(rotation_vector))
    print("Translation Vector:\n {0}".format(translation_vector))
    print("Rotation Matrix: \n {0}".format(rot_mat))
    print("saved world_points, rotation matrix and translation matrix text files \n ")
