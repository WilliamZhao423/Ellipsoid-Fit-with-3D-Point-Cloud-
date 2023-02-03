# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from skimage.draw import ellipsoid
from inner_ellipsoid import inner_ellipsoid_fit
import open3d as o3d # Added @Yimin Zhao
from numpy.linalg import eig
import math 

def outer_ellipsoid_fit(points, tol=0.001):
    """
    Find the minimum volume ellipsoid enclosing (outside) a set of points.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1
    """

    points = np.asmatrix(points)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    u = np.ones(N) / N
    err = 1 + tol
    while err > tol:
        X = Q * np.diag(u) * Q.T
        M = np.diag(Q.T * la.inv(X) * Q)
        jdx = np.argmax(M)
        step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
        new_u = (1 - step_size) * u
        new_u[jdx] += step_size
        err = la.norm(new_u - u)
        u = new_u

    c = u * points  # center of ellipsoid
    A = la.inv(points.T * np.diag(u) * points - c.T * c) / d

    # U, D, V = la.svd(np.asarray(A))
    # rx, ry, rz = 1. / np.sqrt(D)
    #
    # return rx, ry, rz
    return np.asarray(A), np.squeeze(np.asarray(c))


def plot_ellipsoid(A, centroid, color, ax):
    """

    :param A: matrix
    :param centroid: center
    :param color: color
    :param ax: axis
    :return:
    """
    centroid = np.asarray(centroid)
    A = np.asarray(A)
    U, D, V = la.svd(A)
    rx, ry, rz = 1. / np.sqrt(D)
    u, v = np.mgrid[0:2 * np.pi:20j, -np.pi / 2:np.pi / 2:10j]
    x = rx * np.cos(u) * np.cos(v)
    y = ry * np.sin(u) * np.cos(v)
    z = rz * np.sin(v)
    E = np.dstack((x, y, z))
    E = np.dot(E, V) + centroid
    x, y, z = np.rollaxis(E, axis=-1)
    ax.plot_wireframe(x, y, z, cstride=1, rstride=1, color=color, alpha=0.2)
    ax.set_zlabel('Z-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_xlabel('X-Axis')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-10, 50)

if __name__ == '__main__':

    #some random points
    #points = np.array([[ 0.53135758, -0.25818091, -0.32382715],
    #               [ 0.58368177, -0.3286576,  -0.23854156,],
    #               [ 0.18741533,  0.03066228, -0.94294771],
    #               [ 0.65685862, -0.09220681, -0.60347573],
    #               [ 0.63137604, -0.22978685, -0.27479238],
    #               [ 0.59683195, -0.15111101, -0.40536606],
    #               [ 0.68646128,  0.0046802,  -0.68407367],
    #               [ 0.62311759,  0.0101013,  -0.75863324]])
                   
    pcd = o3d.io.read_point_cloud("/home/yimim/Public/pcl_test/NDT/build/10.pcd")
    points = np.asarray(pcd.points)                                 
    A_outer, centroid_outer = outer_ellipsoid_fit(points)
    print("A : ", A_outer)
    print("centroid : ", centroid_outer)

    # Compute the eigenvalues and eigenvectors for matrix A
    w,v=eig(A_outer)
    print('E-value:', w)
    print('E-vector:', v) 
    a = math.sqrt(1/w[0])
    b = math.sqrt(1/w[1])
    c = math.sqrt(1/w[2])
    print("a : ", a)
    print("b : ", b)
    print("c : ", c)
    
    # Save the points to pcd file
    #pcd_t = o3d.t.geometry.PointCloud()
    #pcd_t.point["positions"] = o3d.core.Tensor(points)
    #o3d.t.io.write_point_cloud("pointcloud_p.pcd", pcd_t)
    
    xyzi = np.random.rand(100, 4)
    xyz = xyzi[:,0:3]
    i = [[i] for i in xyzi[:,3]]
    pcd_ = o3d.t.geometry.PointCloud()
    pcd_.point["positions"] = o3d.core.Tensor(xyz)
    pcd_.point["intensities"] = o3d.core.Tensor(i)
    o3d.t.io.write_point_cloud("pointcloud.pcd", pcd_)
     
    
    #B, d = inner_ellipsoid_fit(points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=1, label='OG Points')
    plot_ellipsoid(A_outer, centroid_outer, 'green', ax)
    #plot_ellipsoid(B, d, 'orange', ax)
    plt.legend(loc='best')

    plt.show()
