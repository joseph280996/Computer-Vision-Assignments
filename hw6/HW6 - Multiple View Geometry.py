# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import cv2
import numpy as np
import requests
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize
import os

matplotlib.rcParams['figure.figsize'] = (12,8)

# %% [markdown]
# # HW6: Multiple View Geometry
#
# In the lst couple of weeks we've talked about multiple-view geometry. The main pursuits in this domain are:
# * Reconstructing the 3D geometry of the objects in the visible scene
#   * Dense reconstruction with stereo
#   * Sparse reconstruction with feature key-points (e.g corners)
# * Estimating the pose (location, orientation) of the cameras looking at the scene
#

# %% [markdown]
# We will work with images from a well known Structure-from-Motion dataset: https://cvlab.epfl.ch/data/data-strechamvs/
#
# Download images

# %%
DOWNLOAD_IDS = [
    '1G9NS27L2YEgMUzU34iquIm-A5MBlYiwr',
    '1gR65HLkcAcksolu3cU46MxPY4cn_V8Mw',
    '1CLgiVSltl69dYmZOQV9Zt7wfWFTAochn',
    '1WiuVyg4btUJQU_1XOhAOwpYWjlATuyaw',
    '1XYWtRQJ6REJEpEgza0tEoIXNNLTB2k88',
    '1661OnKt8Ns5KvWiy6FSI95G7NXVnKOCh',
    '177YlElwbg8vlPiLzk5P-JKD4Ce3a7vs7',
    '1Ucr5vDIzUwdaxde5f_gdk2FYH3jl8suV',
    '1M6j1fdKcgULUFqhy6kCVwOQUOFtzref7',
    '1CQpPpefDgxv5DDGYprmvlNK_cq-hNqDi',
    '1Bx3AJo9q_ttZ-qv3ZZT1B_9lJeO5kguz',
    ]
images = []
for dl_id in DOWNLOAD_IDS:
    filename = f"image{dl_id}.jpg"
    if not os.path.exists(filename):
        with open(filename, "wb") as f:
            f.write(requests.get("https://drive.google.com/uc?id=%s"%(dl_id)).content)
    images.append(cv2.resize(cv2.imread(filename, cv2.IMREAD_COLOR)[...,::-1], (0,0), fx=0.25, fy=0.25))


# %% [markdown]
# ### Image Feature Graph
# To get started with an MVG pipeline we need to extract features and descriptors from images.
#
# Below is a class that I wrote to help with some of this work so you can focus on the algorithms.

# %%
# The MatchMaker class is a helper class to store the keypoints matches and match graph
# and provide some helper functions to get robust, aligned matches between two images.
# I'm providing this class to you to speed up our process, but you should read through
# it to understand how it works roughly.
class MatchMaker:
    def __init__(self) -> None:
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50))
        self.images = []
        self.kpts = []
        self.descs = []
        self.matches = {}
        self.kpts_match_graph = None
        self.point3d_camera_visibility = None
        self.map_3d = None
        self.poses = None
        pass

    def getMatchesFilterFundamental(self, left_image_index, right_image_index):
        matches_raw = list(self.matcher.knnMatch(
            self.descs[left_image_index], self.descs[right_image_index], 2))
        matches_ = []
        for (m, n) in matches_raw:
            if m.distance < 0.7*n.distance:
                matches_.append(m)

        # filter by finding the fundamental matrix with RANSAC
        mptsif, mptsjf = zip(*[(self.kpts[left_image_index][m.queryIdx].pt,
                                self.kpts[right_image_index][m.trainIdx].pt)
                               for m in matches_])
        mptsif, mptsjf = np.array(mptsif), np.array(mptsjf)
        _, mask = cv2.findFundamentalMat(mptsif, mptsjf, cv2.FM_RANSAC)
        matches_ = [matches_[i] for i in range(len(matches_)) if mask[i]]

        return np.array(matches_)

    def addImagesAndExtractKeypoints(self, images):
        self.images = images
        self.kpts, self.descs = zip(
            *[self.detector.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None) for img in images])
        # store the 3D points visibility graph in a 2D array
        self.point3d_camera_visibility = -np.ones((len(self.images), 500), np.int32)
        # camera poses
        self.poses = np.zeros((len(self.images), 3, 4))

    def buildMatchGraph(self):
        # Match every image with every other image (without repetition).
        # For each pair of images, get the matches and fundamental matrix using the helper function.
        # store in a dictionary with key (i,j) and value (mptsif,mptsjf,inliers,f_ij)
        self.matches = {}
        # store the match graph in a 2D array
        self.kpts_match_graph = -np.ones((len(self.images), len(self.images), np.max(
            [len(kptsi) for kptsi in self.kpts])), dtype=np.int32)
        
        for i, j in product(range(len(images)), repeat=2):
            if i < j:
                self.matches[(i, j)] = self.getMatchesFilterFundamental(i, j)

                # update the match graph kpts_match_graph
                for m in self.matches[(i, j)]:
                    self.kpts_match_graph[i, j, m.queryIdx] = m.trainIdx # right view index
                    self.kpts_match_graph[j, i, m.trainIdx] = m.queryIdx # left view index

    def getMatchGraph(self):
        if self.kpts_match_graph is None:
            raise Exception("The match graph is not computed yet.")
        return self.kpts_match_graph

    def aligned2D(self, left_image_index, right_image_index):
        ptsl2d, ptsr2d = zip(*[(self.kpts[left_image_index][m.queryIdx].pt,
                                self.kpts[right_image_index][m.trainIdx].pt)
                               for m in self.matches[(left_image_index, right_image_index)]])
        return np.array(ptsl2d), np.array(ptsr2d)
    
    def aligned2DNotInMap(self, left_image_index, right_image_index):
        ptsl2d, ptsr2d, backidxL, backidxR = zip(*[(self.kpts[left_image_index][m.queryIdx].pt,
                                                    self.kpts[right_image_index][m.trainIdx].pt,
                                                    m.queryIdx, 
                                                    m.trainIdx)
                               for m in self.matches[(left_image_index, right_image_index)]
                                 if not m.queryIdx in self.point3d_camera_visibility[left_image_index] and 
                                    not m.trainIdx in self.point3d_camera_visibility[right_image_index]
                               ])
        return np.array(ptsl2d), np.array(ptsr2d), np.array(backidxL), np.array(backidxR)
    
    def alignedIndices(self, left_image_index, right_image_index):
        indl2d, indr2d = zip(*[(m.queryIdx, m.trainIdx)
                               for m in self.matches[(left_image_index, right_image_index)]])
        return np.array(indl2d), np.array(indr2d)

    def addNewPoints3D(self, pts3d, li, rj, mask=None):
        if self.map_3d is None:
            self.map_3d = np.zeros((0, 3))
        if mask is None:
            mask = np.ones(pts3d.shape[0], dtype=np.bool)

        mapIS = self.map_3d.shape[0]
        mapIE = mapIS + pts3d[mask].shape[0]
        alignedIdxL, alignedIdxR = self.alignedIndices(li, rj)
        self.point3d_camera_visibility[li, mapIS:mapIE] = alignedIdxL[mask]
        self.point3d_camera_visibility[rj, mapIS:mapIE] = alignedIdxR[mask]
        self.map_3d = np.concatenate([self.map_3d, pts3d[mask]])

    def R(self, i):
        return self.poses[i, :3, :3]
    
    def t(self, i):
        return self.poses[i, :3, 3]
    
    # get 2D keypoints for a given image (rj, "right") which match 3D points in 
    # the map from another image (li, "left")
    def alignedKptsTo3DMap(self, li_3d, rj_2d):
        # left view indices from 3D points on the current map
        li_3d_idx = self.point3d_camera_visibility[li_3d]
        # right view indices that align to the left view indices above
        indices_rj = self.kpts_match_graph[li_3d, rj_2d, li_3d_idx]
        # get the keypoints from the right view
        selected_rj_kpts = np.array(self.kpts[rj_2d])[indices_rj[indices_rj > -1]]
        # get the keypoints from the left view
        selected_li_kpts = np.array(self.kpts[li_3d])[li_3d_idx[li_3d_idx > -1]]
        return selected_li_kpts, selected_rj_kpts

    # get 2D points for a given image (rj, "right") with corresponding 3D 
    # points in the map from any other image (li, "left")
    def aligned2D3D(self, rj_2d):
        mpts2Drj = []
        mpts3D = []
        backmapping = []

        # for each 3D point in the map
        for p3d_id in range(self.map_3d.shape[0]):
            # check its visibility in all the left views
            for li_3d in range(self.point3d_camera_visibility.shape[0]):
                if li_3d != rj_2d: # skip the right view...
                    # the index of the 3D point in the left view
                    li_3d_kpt_idx = self.point3d_camera_visibility[li_3d, p3d_id]
                    if li_3d_kpt_idx < 0: 
                        continue # this 3D point is not visible in the left view
                    rj_2d_kpt_idx = self.kpts_match_graph[li_3d, rj_2d, li_3d_kpt_idx]
                    if rj_2d_kpt_idx > -1: # this 3D point is visible in both views
                        pt = self.kpts[rj_2d][rj_2d_kpt_idx].pt
                        mpts2Drj.append((pt[0], pt[1], 1.0))
                        mpts3D.append(self.map_3d[p3d_id])
                        backmapping.append((p3d_id, rj_2d_kpt_idx))
                        break
                        
        # return an interleaved array of 2D and 3D points
        return np.stack([mpts2Drj, mpts3D], axis=1), backmapping
    
    def alignedMapTo2DAndVisibility(self):
        # return an aligned list of 2D image points and a list of 3D points from the map
        mpts2DForViews = np.zeros((self.point3d_camera_visibility.shape[0], self.map_3d.shape[0], 2), dtype=np.float32)
        visibility = np.zeros((self.point3d_camera_visibility.shape[0], self.map_3d.shape[0]), dtype=np.uint8)

        # for each 3D point in the map
        for p3d_id in range(self.map_3d.shape[0]):
            # check its visibility in all the left views
            for view_i in range(self.point3d_camera_visibility.shape[0]):
                # the index of the 3D point in the view
                li_3d_kpt_idx = self.point3d_camera_visibility[view_i, p3d_id]
                if li_3d_kpt_idx < 0: 
                    # this 3D point is not visible in the view
                    mpts2DForViews[view_i, p3d_id] = [0, 0]
                    visibility[view_i, p3d_id] = 0
                else:
                    # this 3D point is visible in the view
                    pt = self.kpts[view_i][li_3d_kpt_idx].pt
                    mpts2DForViews[view_i, p3d_id] = [pt[0], pt[1]]
                    visibility[view_i, p3d_id] = 1

        return self.map_3d, mpts2DForViews, visibility


# %%
# initialize the MatchMaker class and extract keypoints from the images
mm = MatchMaker()
mm.addImagesAndExtractKeypoints(images)
mm.buildMatchGraph()

# %% [markdown]
# Initialize the camera intrinsic matrix. This is given to us, and guranteed to be similar and cosistent across images of the dataset.

# %%
# K is the camera matrix, given to us by the dataset provider
K_original = np.array([[2759.48, 0, 1520.69],[0, 2764.16, 1006.81],[0, 0, 1]])
# W and H are the width and height of the images
W,H = mm.images[0].shape[1], mm.images[0].shape[0]
# K_s is the scaled camera matrix, we use it because we scaled the images to 0.25 their original size
K, _ = cv2.getOptimalNewCameraMatrix(K_original, np.zeros(5), (W*4,H*4), 0, (W,H))
# Kinv is the inverse of K
Kinv = np.linalg.inv(K)

# %%
# show the match graph for all images
plt.figure(figsize=(10, 10))
plt.imshow(np.vstack(mm.getMatchGraph()[:]) > -1)
plt.show()

# %%
# draw the matches between image 0 and image 1
plt.imshow(cv2.drawMatches(mm.images[0],mm.kpts[0],mm.images[1],mm.kpts[1],mm.matches[(0,1)],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
plt.title('image 0 -> 1: # points %d'%(len(mm.matches[(0,1)])));

# %%
# draw the matches between image 3 and image 7
plt.imshow(cv2.drawMatches(mm.images[3],mm.kpts[3],mm.images[7],mm.kpts[7],mm.matches[(3,7)],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
plt.title('image 3 -> 7: # points %d'%(len(mm.matches[(3,7)])));

# %%
# plot the number of inliers between images 0, 1 and 2 and the rest of the images
# in a multi-bar plot
N = len(mm.images)
plt.bar(np.arange(1,N)-0.11,[len(mm.matches[(0,i)]) for i in range(1,N)], label='image 0', width=0.33)
plt.bar(np.arange(1,N)+0.22,[0]+[len(mm.matches[(1,i)]) for i in range(2,N)], label='image 1', width=0.33)
plt.bar(np.arange(1,N)+0.55,[0,0]+[len(mm.matches[(2,i)]) for i in range(3,N)], label='image 2', width=0.33)
plt.legend()
plt.title('number of inliers');


# %% [markdown]
# Notice the sharp drop in inlier matches as the views change the perspective. Only consecutive images have a high inlier count.
#
# However - we do want to work with a wide baseline. That's because very small shifts between viewpoints can approximate a Homography ("flat" planar) transformation, which is NOT what we want.

# %% [markdown]
# Here comes your part!

# %% [markdown]
# ## 1. Two-frame Structure-from-Motion
#
# 1. Estimate essential matrix (with re-scaling)
# 1. Decompose to find $[R|t]$
# 1. Triangulate sparse 3D point cloud

# %% [markdown]
# ### Estimate the essential matrix - the linear method
#
# The epipolar constraint: $x_\mathrm{R}^\top E x_\mathrm{L} = 0$ leads to the following $Ab=0$ system of equations: (Szeliski's eqn 11.33)
# $$
# \begin{pmatrix}x_R & y_R & 1 \end{pmatrix}
# \begin{bmatrix}e_{00} & e_{01} & e_{02} \\ e_{10} & e_{11} & e_{12} \\ e_{20} & e_{21} & e_{22}\end{bmatrix}\begin{pmatrix}x_L\\y_L\\1 \end{pmatrix} = 0
# \\
# \begin{pmatrix}
# x_Re_{00} + y_Re_{10} + e_{20} &
# x_Re_{01} + y_Re_{11} + e_{21} &
# x_Re_{02} + y_Re_{12} + e_{22} 
# \end{pmatrix}
# \begin{pmatrix}x_L\\y_L\\1 \end{pmatrix} = 0
# \\
# x_Lx_Re_{00} + x_Ly_Re_{10} + x_Le_{20} + 
# y_Lx_Re_{01} + y_Ly_Re_{11} + y_Le_{21} +
# x_Re_{02} + y_Re_{12} + e_{22} 
# = 0
# \\
# \begin{pmatrix}
# \cdots \\
# x_Lx_R & y_Lx_R & x_R &
# x_Ly_R & y_Ly_R & y_R &
# x_L & y_L & 1 \\
# \cdots \\
# \end{pmatrix}
# \begin{pmatrix}
# e_{00} \\ e_{01} \\ e_{02} \\ e_{10} \\ e_{11} \\ e_{12} \\ e_{20} \\ e_{21} \\ e_{22} \\
# \end{pmatrix}=0
# $$
# (where every row of A is essentially $x_\mathrm{R}^\top x_\mathrm{L}$ ($3\times3$) flattened, e.g. `(xR.T @ xL).ravel()`)
# $$
# \begin{pmatrix}x_R \\ y_R \\ 1 \end{pmatrix}\begin{pmatrix}x_L & y_L & 1 \end{pmatrix}=
# \begin{bmatrix}x_Lx_R & y_Lx_R & x_R \\ x_Ly_R & y_Ly_R & y_R  \\ x_L & y_L & 1  \end{bmatrix}
# $$
#
# However, remember we talked about the scaling problem, where $x*x$ is orders of magnitude larger than $x$ and $1$, therefore we should normalize the points to the $[-1,1]$ range before solving. Afterwards we can apply the inverse scaling to $E$ to negate this effect.
#
# Populate the matrix $A$ and solve for $E$ in the least squares sense: $\hat{b} = \mathop{\arg\min}_b|Ab|^2$, which means taking the SVD (`np.linalg.svd`) and using the last row of $V^\top$, where the singular value is 0.

# %%
# get an aligned list of 2D image points from view 3 and 7
mpts01o,mpts10o = mm.aligned2D(3, 7) # `o` is for "original"

# Transform the points to normalized coordinates - meaning multiply on the left by the inverse of the camera matrix
# use cv2.convertPointsToHomogeneous to convert the points to homogeneous coordinates
# due to some weirdness in the way cv2 works, we need to use np.squeeze to remove the extra dimension
# as well as .T (transpose) to get the correct shape (N,3) instead of (N,1,3)
# the whole thing looks like this: np.matmul(Kinv, cv2.convertPointsToHomogeneous(pts).squeeze().T).T
mpts01f,mpts10f = np.matmul(Kinv, cv2.convertPointsToHomogeneous(mpts01o).squeeze().T).T,np.matmul(Kinv, cv2.convertPointsToHomogeneous(mpts10o).squeeze().T).T


# %%
# Calculate the Essential Matrix according to the above equations
def calculateEssentialMatrix(ptsLeftHomog, ptsRightHomog):
    # Ensure ptsLeftHomog and ptsRightHomog are numpy arrays
    ptsLeftHomog = np.asarray(ptsLeftHomog)
    ptsRightHomog = np.asarray(ptsRightHomog)

    # Number of points
    num_points = ptsLeftHomog.shape[0]

    # Construct the A matrix using np.kron
    A = np.zeros((num_points, 9))
    for i in range(num_points):
        A[i] = np.kron(ptsRightHomog[i], ptsLeftHomog[i])

    # Solve Ab=0 using SVD
    _, _, V = np.linalg.svd(A)

    # Take E as the last row of VT (last column of V)
    E = V[-1].reshape((3, 3))

    # Enforce the rank-2 constraint on E using SVD
    U, _, VT = np.linalg.svd(E)
    Sigma = np.diag([1, 1, 0])
    E = np.dot(U, np.dot(Sigma, VT))

    return E



# %% [markdown]
# We will implement a "poor man's RANSAC", randomly selecting 9 point-pairs and calculating $E$ from them, then counting the supporting "inlier" points by checking how far they are from their corresponding epilines.
#
# We will be using our homebrew RANSAC again further down.

# %%
# RANSAC is one of the most popular methods for finding the best model fit for a given set of data
# In this case, we want to find the best Essential Matrix for the given set of 2D image points
# We will use the following steps:
# 1. Randomly select 9 points from the set of 2D image points
# 2. Calculate the Essential Matrix for these 9 points with the above function (calculateEssentialMatrix)
# 3. Find the inliers for this Essential Matrix
# 4. Repeat steps 1-3 15000 times and keep the best model (the one with the most inliers)
# 5. Recalculate the Essential Matrix for the inliers of the best model

maxv = 0
best_inliers = None 
best_E = None

total_points = mpts01f.shape[0]

for i in range(15000):
    # sample 9 points randomly from mpts01f and calculate `E_guess` (with calculateEssentialMatrix)
    # then use `E_guess` to find the inliers by calculating the epipolar distance
    # e.g. your epilines would be l0 = E_guess @ mpts01f.T  (this will be a 3xN matrix)
    # then you can calculate the epipolar distance d for each point by doing
    # e.g. [np.matmul(mpts10f[i], l) for i,l in enumerate(l0)]
    # and then you can find the inliers by e.g. d < 0.0002
    # recalculate the E matrix with the inliers by "masking" the points
    # keep a "score" for the number of inliers and find the best model
    # save the best model in the variable `E`

    # Sample 9 points randomly from mpts01f and calculate `E_guess`
    idx = np.random.choice(mpts01f.shape[0], size=9, replace=False)
    pts_left = mpts01f[idx]
    pts_right = mpts10f[idx]
    E_guess = calculateEssentialMatrix(pts_left, pts_right)

    # Find the inliers by calculating the epipolar distance
    l0 = E_guess @ mpts01f.T
    d = np.array([np.abs(np.matmul(mpts10f[i], l)) for i, l in enumerate(l0.T)])
    inliers = d < 0.0002

    # Recalculate the E matrix with the inliers by "masking" the points
    inlier_pts_left = mpts01f[inliers]
    inlier_pts_right = mpts10f[inliers]
    E_inliers = calculateEssentialMatrix(inlier_pts_left, inlier_pts_right)

    # Keep a "score" for the number of inliers and find the best model
    score = np.sum(inliers)
    if score > maxv:
        maxv = score
        best_inliers = inliers
        best_E = E_inliers

        # Calculate the error for the best model
        l1 = np.matmul(best_E, mpts01f.T)
        d1 = np.array([np.abs(np.matmul(mpts10f[i], l)) for i, l in enumerate(l1.T)])
        error = np.mean(d1[best_inliers])

        # Print the desired statement
        print(f"{maxv} / {total_points}. error = {error}")

# Save the best model in the variable `E`
E = best_E

# %%
E

# %%
# Use cv2.findEssentialMat to verify your result
E_cv, mask = cv2.findEssentialMat(mpts01f[:,:2],mpts10f[:,:2],np.eye(3,3),method=cv2.RANSAC,prob=0.999,threshold=0.0002)
E_cv, cv2.countNonZero(mask)

# %%
# if your result is widely different from the cv2 result, check your work.

# %% [markdown]
# ### Epipolar Lines
#
# Let's inspect the epipolar lines arising from the essential matrix you've found. Verifying the epilines make sense is a great way to overall make sure your solution is good and matches your expectation (i.e the motion between cameras).

# %%
# find the fundamental matrix from the essential matrix by using the camera matrix: F = Kinv.T * E * Kinv
# you can optionally normalize the fundamental matrix by dividing it by its F[2,2] element
# save it in the variable `F_fromE`

F = np.matmul(np.matmul(Kinv.T, E), Kinv)

F_fromE = F / F[2,2]


# %% [markdown]
# Calculate the epipolar lines, which is simply $l_\mathrm{Left} = F \tilde{x}_\mathrm{Right}$, where $\tilde{x}$ is the homogeneous augmented 2D point $x$. Each point in the right image is a line on the left image, w.l.o.g

# %%
# compute the epipolar lines from the F matrix, just like you've done before in the previous 
# section with the essential matrix.
# use cv2.convertPointsToHomogeneous to convert pts_list to homogeneous coordinates
def computeEpipolarLines(F, pts_list):
    pts_homogeneous = cv2.convertPointsToHomogeneous(pts_list).squeeze().T
    epipolar_lines = np.matmul(F, pts_homogeneous)
    return epipolar_lines.T



# %%
# for your convenience, we've provided the code to draw the epipolar lines.
# you can use it to verify your work and that the epipolar lines match what we got.
lines_right_image = computeEpipolarLines(F_fromE, mpts01o) # left points -> lines in right image
lines_left_image = computeEpipolarLines(F_fromE.T, mpts10o) # right points -> lines in left image

# draw epipolar lines
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(images[3])
step = 10
for i in range(0, len(lines_left_image), step):
    a,b,c = lines_left_image[i]
    x0,y0 = 0, int(-c/b)
    x1,y1 = W, int((-a*W-c)/b)
    plt.plot([x0,x1],[y0,y1],color='r')
    plt.scatter([mpts01o[i,0]],[mpts01o[i,1]],c='b')
plt.axis('off')
plt.ylim(H, 0)

plt.subplot(1,2,2)
plt.imshow(images[7])
for i in range(0, len(lines_right_image), step):
    a,b,c = lines_right_image[i]
    x0,y0 = map(int, [0, -c/b])
    x1,y1 = map(int, [W, -(a*W+c)/b])
    plt.plot([x0,x1],[y0,y1],color='b')
    plt.scatter([mpts10o[i,0]],[mpts10o[i,1]],c='r')
# remove the axes
plt.axis('off')
# crop to just the image (no whitespace)
plt.ylim(H, 0)

plt.tight_layout()

# %% [markdown]
# Verify that your epilines make sense, e.g. that they convrge in a point in the right side of the image but outside of the image - that's where the other camera would be. The "right" image will have the other camera on the left, and vice versa.

# %% [markdown]
# ### Decompose $E$ to $[R|t]$
# Recall the essential matrix is composed: $E = [t]_\times R$.
#
# The decomposition can be performed using SVD, e.g. $E = U\Sigma V^\top$, and set $t$ to be the last column of $\pm U$, while
# $$
# \begin{align}
# W &= \begin{bmatrix}0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1\end{bmatrix}  \mathrm {,\,\,A\,\, 90^\circ\,\, rotation}\\
# R'_1 &= UWV^\top\\
# R'_2 &= UW^{-1}V^\top\\
# R'_3 &= -UWV^\top\\
# R'_4 &= -UW^{-1}V^\top\\
# \end{align}
# $$
# But we keep only the 2 rotation matrices e.g. $R_1, R_2$ that have positive determinant.
#
# This results in 4 configurations: $(t,R_1)$,$(t,R_2)$,$(-t,R_1)$,$(-t,R_2)$. See your readings for the reason.
#
# To find the correct pair we should use the "cheirality check", which essentially means we triangulate 3D points and check they have positive Z coordinates and they are indeed in front of both cameras.

# %%
# Take SVD of E
# t is the last column of U
# there are four possible rotations: R1 = UWV^T, R2 = UW^TV^T, R3 = -UWV^T, R4 = -UW^TV^T
# but keep only the 2 rotations with positive determinant (use np.linalg.det)
# keep the good rotations in a variable called Rs

# Take SVD of E
U, _, VT = np.linalg.svd(E)

# Extract the last column of U as the translation vector (t)
t = U[:, -1]

# Define the W matrix
W = np.array([[0, -1, 0],
              [1, 0, 0],
              [0, 0, 1]])

# Compute the four possible rotations
R1 = np.matmul(np.matmul(U, W), VT)
R2 = np.matmul(np.matmul(U, W.T), VT)
R3 = np.matmul(np.matmul(-U, W), VT)
R4 = np.matmul(np.matmul(-U, W.T), VT)

# Keep only the rotations with positive determinant
rotations = [R1, R2, R3, R4]
Rs = []
for R in rotations:
    if len(Rs) == 2:
        break
    if np.linalg.det(R) > 0:
        Rs.append(R)

# %%
# print out your results
Rs,t

# %%
# and you can cross-check your result with cv2.decomposeEssentialMat
cv2.decomposeEssentialMat(E)


# %%
# again if you got a vastly different result, check your work
# but small changes in the domain of floating point errors are fine (e.g. 1e-3 range)

# %% [markdown]
# ### Triangulate 3D points
# Recall our work on triangulation once we have found the $R,t$ parameters:
# $$
# \displaystyle
# \begin{align}
# \begin{bmatrix}\lambda x^{(l)}\\\lambda y^{(l)}\\\lambda\end{bmatrix}
# &=
# \begin{bmatrix}
# f_x & 0 & c_x \\
# 0 & f_y & c_y \\
# 0 & 0 & 1 \\
# \end{bmatrix}
# \begin{bmatrix}
# r_1 & r_2 & r_3 & t_x\\
# r_4 & r_5 & r_6 & t_y\\
# r_7 & r_8 & r_9 & t_z
# \end{bmatrix}
# \begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}
# =
# \begin{bmatrix}
# p_{11} & p_{12} & p_{13} & p_{14} \\
# p_{21} & p_{22} & p_{23} & p_{24} \\
# p_{31} & p_{32} & p_{33} & p_{34} \\
# \end{bmatrix}
# \begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}
# \\
# \begin{bmatrix}\lambda x^{(r)}\\\lambda y^{(r)}\\\lambda\end{bmatrix}
# &=
# \begin{bmatrix}
# f_x & 0 & c_x & 0\\
# 0 & f_y & c_y & 0\\
# 0 & 0 & 1 & 0\\
# \end{bmatrix}
# \begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}\\
# \end{align}
# $$
# Expand
# $$
# \begin{align}
# x^{(l)}p_{31}X + x^{(l)}p_{32}Y + x^{(l)}p_{33}Z + x^{(l)}p_{34} &= p_{11}X + p_{12}Y + p_{13}Z + p_{14}\\
# y^{(l)}p_{31}X + y^{(l)}p_{32}Y + y^{(l)}p_{33}Z + y^{(l)}p_{34} &= p_{21}X + p_{22}Y + p_{23}Z + p_{24}\\
# x^{(r)}Z &= f_x X + c_x Z \\
# y^{(r)}Z &= f_y Y + c_y Z
# \end{align}
# $$
# Rearrange
# $$
# \begin{align}
# (x^{(l)}p_{31} - p_{11})X + (x^{(l)}p_{32} - p_{12})Y + (x^{(l)}p_{33} - p_{13})Z + x^{(l)}p_{34} - p_{14} &= 0\\
# (y^{(l)}p_{31} - p_{21})X + (y^{(l)}p_{32} - p_{22})Y + (y^{(l)}p_{33} - p_{23})Z + y^{(l)}p_{34} - p_{24} &= 0\\
# -f_x X + 0Y + (x^{(r)} - c_x)Z &= 0 \\
# 0X + -f_y Y + (y^{(r)} - c_y)Z &= 0 \\
# \end{align}
# $$
# Matrix form
# $$
# \begin{bmatrix}
# x^{(l)}p_{31} - p_{11} & x^{(l)}p_{32} - p_{12} & x^{(l)}p_{33} - p_{13} & x^{(l)}p_{34} - p_{14} \\
# y^{(l)}p_{31} - p_{21} & y^{(l)}p_{32} - p_{22} & y^{(l)}p_{33} - p_{23} & y^{(l)}p_{34} - p_{24} \\
# -f_x & 0 & x^{(r)} - c_x \\
# 0 & -f_y & y^{(r)} - c_y
# \end{bmatrix}
# \begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}=0
# $$
#
# Now if we have 2 contributions to this system we can solve this linear system of equations $Ax=0$ in the constrained least squares sense (`np.linalg.svd`, take last row ot $V^\top$).

# %% [markdown]
# Write the triangulation routine

# %%
# write a function that triangulates points given a "right" camera extrinsics (R,t)
# assume that the "left" camera intrinsics are the identity matrix
# first compute the projection matrix P = K[R|t] (use e.g. np.matmul)
# for every point:
#   populate the A matrix according to the above equations
#   use np.linalg.svd to compute the SVD of A
#   the 3D point is the last column of V (last row of V^T)
#   normalize the homogeneous 3D point by dividing by its last element (homogeneous coordinate divide)
# return the 3D points in a numpy array of shape (N,3)
def triangulatePoints(pts2Dr, pts2Dl, R, t, K_):
    # Compute the projection matrix P = K[R|t]
    Rt = np.concatenate((R, t.reshape(3, 1)), axis=1)
    P = np.matmul(K_, Rt)

    # Number of points
    N = pts2Dr.shape[0]

    # Create an array to store the triangulated 3D points
    pts3D = np.zeros((N, 3))

    for i in range(N):
        # Extract the corresponding 2D points
        x_r, y_r = pts2Dr[i]
        x_l, y_l = pts2Dl[i]

        # Populate the A matrix
        A = np.array([
            [x_r * P[2, 0] - P[0, 0], x_r * P[2, 1] - P[0, 1], x_r * P[2, 2] - P[0, 2], x_r * P[2, 3] - P[0, 3]],
            [y_r * P[2, 0] - P[1, 0], y_r * P[2, 1] - P[1, 1], y_r * P[2, 2] - P[1, 2], y_r * P[2, 3] - P[1, 3]],
            [x_l * Rt[2, 0] - Rt[0, 0], x_l * Rt[2, 1] - Rt[0, 1], x_l * Rt[2, 2] - Rt[0, 2], x_l * Rt[2, 3] - Rt[0, 3]],
            [y_l * Rt[2, 0] - Rt[1, 0], y_l * Rt[2, 1] - Rt[1, 1], y_l * Rt[2, 2] - Rt[1, 2], y_l * Rt[2, 3] - Rt[1, 3]]
        ])

        # Compute the SVD of A
        _, _, VT = np.linalg.svd(A)

        # The 3D point is the last column of V (last row of V^T)
        pt3D_homogeneous = VT[-1]

        # Normalize the homogeneous 3D point
        pt3D = pt3D_homogeneous[:3] / pt3D_homogeneous[3]

        # Store the 3D point in the array
        pts3D[i] = pt3D

    return pts3D


# %% [markdown]
# Make a decision about the 4 possible configurations (e.g. $(t,R_1)$,$(t,R_2)$,$(-t,R_1)$,$(-t,R_2)$) using the following criteria:
# 1. Cheirality check: Count how many points are in front of the camera (positive +z coordinate)
# 1. Reprojection check: Distance between original 2D point and 3D point reprojected back to 2D

# %%
# since we have two possible rotations and two possible translations (4 possible solutions)
# we need to pick the best one. We can do this by checking the cheirality of the points
# and the reprojection error. The best solution is the one with the most points in front
# of the camera (Z > 0) and the smallest reprojection error.
# write a loop that tries all 4 possible solutions and pick the best one.
# the solutions are: (Rs[0],t), (Rs[0],-t), (Rs[1],t), (Rs[1],-t)
# for each solution:
#   triangulate the points with triangulatePoints you wrote above
#   reproject the points back to the image domain using cv2.projectPoints
#   compute the cheirality (number of points with Z > 0) and the reprojection error
#   compute a score as the ratio of cheirality to reprojection error
#   print out the cheirality, reprojection error and score for each solution
#   keep the best score and the corresponding rotation and translation
# when using cv2.projectPoints, if you're projecting on the left image, use R = np.eye(3), t = np.zeros((3,1))
# if you're projecting on the right image, use R = Rs[i], t = t
# to calculate the reprojection error, use np.linalg.norm to compute the norm of the difference between the
# reprojected points (from cv2.projectPoints) and the original points (e.g. mpts01o for the left image)
maxv = 0
R_best = t_best = None
# for rot,tra in ...

# %% [markdown]
# The reprojection error above should be < 10. If that's not the case it's likely the $E$ matrix isn't right - try finding it again.

# %%
R_best, t_best

# %%
# verify your result vs cv2.recoverPose
_,R_cv,t_cv,_,_ = cv2.recoverPose(E, mpts01o, mpts10o, K, distanceThresh=3.0)
R_cv,t_cv

# %%
# as always - if your result is vastly different, check your work.
# you should expect to get a result that is very very close to the cv2.recoverPose result

# %% [markdown]
# At this point we should start building our 3D map, saying for every 3D point which 2D views and points support it

# %%
# triangulate points using the best rotation and translation
pts3d = triangulatePoints(mpts01o, mpts10o, R_best, t_best, K)

# since this has to do with the match graph - i provide the code for this part

# reproject points back to image domain
projPts2dLeft,_  = cv2.projectPoints(pts3d, (0,0,0), (0,0,0), K, None)
projPts2dRight,_ = cv2.projectPoints(pts3d, R_best, t_best, K, None)

# only keep points that have a small reprojection error (less than 5 pixels)
mask_reproj = np.all(np.dstack([
    np.linalg.norm(projPts2dRight.squeeze() - mpts10o, axis=1) < 5, 
    np.linalg.norm(projPts2dLeft.squeeze() - mpts01o, axis=1) < 5]), axis=2).squeeze()

# update the 3D point map, and visibility graph with the visibility of each point.
# currently the 3D points are visible from only two images: 3 and 7
mm.addNewPoints3D(pts3d, 3, 7, mask_reproj)
mm.poses[3] = np.hstack([np.eye(3), np.zeros((3,1))])
mm.poses[7] = np.hstack([R_best, t_best])

# repoject the points back to the image domain again after filtering for visualization
projPts2dLeft,_  = cv2.projectPoints(mm.map_3d, mm.R(3), mm.t(3), K, None)
projPts2dRight,_ = cv2.projectPoints(mm.map_3d, mm.R(7), mm.t(7), K, None)

# %%
# the visibility matrix
plt.figure(figsize=(20,10))
plt.imshow((mm.point3d_camera_visibility > 0)[:,:100])

# %% [markdown]
# Show the points reprojected alongside the originals, any strong deviation here will suggest a bug

# %%
# again here's provided code to visualize the 3D points and the reprojection error
# use this to verify that your code is working correctly
%matplotlib inline
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(images[3])
plt.scatter(mpts01o[:,0],mpts01o[:,1],label='Original 2D', s=20)
plt.scatter(projPts2dLeft[:,0,0],projPts2dLeft[:,0,1],label='Reprojected 3D',s=3)
plt.legend()

plt.subplot(1,2,2)
plt.imshow(images[7])
plt.scatter(mpts10o[:,0],mpts10o[:,1],label='Original 2D', s=20)
plt.scatter(projPts2dRight[:,0,0],projPts2dRight[:,0,1],label='Reprojected 3D',s=3)
plt.legend()
plt.tight_layout()

# %%
plt.scatter(pts3d[:,0],pts3d[:,2])
plt.xlabel('X')
plt.ylabel('Z')
plt.title('X-Z plane (view from above)');

# %% [markdown]
# We can see the wall and the fountain basin coming out of it.

# %% [markdown]
# ---
# ## 2. Incremental SfM
# Add another camera to your reconstructed scene: 
# 1. Find matching 2D-3D points
# 1. Find camera pose with linear pose estimation
# 1. Triangulate additional 3D points with the new camera pose

# %% [markdown]
# ### 2D-3D correspondences
# Let's add the view #5. To find camera pose we need to get 2D-3D correspondences.
# To get correspondences we go back to the original matching (Image 3 $\leftrightarrow$ Image 7), and select the 2D points in Image 3 (which created 3D points with Image 7) intersected with 2D points in Image 5.

# %%
# let's get an aligned set of points from the graph and visualize them
selected_3_kpts, selected_5_kpts = mm.alignedKptsTo3DMap(3, 5)

img_5 = cv2.drawKeypoints(images[5], selected_5_kpts, None, color=(0, 255, 0), flags=0)
img_3 = cv2.drawKeypoints(images[3], selected_3_kpts, None, color=(0, 255, 0), flags=0)

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(img_5)
plt.title('Image 5')
plt.subplot(1,2,2)
plt.imshow(img_3)
plt.title('Image 3')
plt.tight_layout()


# %% [markdown]
# seeing the two views and their points make sense is an important step for verification.

# %% [markdown]
# ### Camera pose - the linear method
# Remember, as per usual we start from
# $$
# \lambda\begin{bmatrix}x\\y\\1\end{bmatrix}
# =
# \begin{bmatrix}f_x & 0 & c_x\\0 & f_y & c_y \\0 & 0 & 1\end{bmatrix}
# \begin{bmatrix}
# r_1 & r_2 & r_3 & t_x\\
# r_4 & r_5 & r_6 & t_y\\
# r_7 & r_8 & r_9 & t_z\\
# \end{bmatrix}
# \begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}
# $$
# Multiply on left with $K^{-1}$ (essentially use normalized coordinates) and rearrange (after finding $\lambda$):
# $$
# \displaystyle
# \begin{bmatrix}
# x'(r_7X+r_8Y+r_9Z+t_z)\\
# y'(r_7X+r_8Y+r_9Z+t_z)
# \end{bmatrix}
# =
# \begin{bmatrix}
# r_1 & r_2 & r_3 & t_x\\
# r_4 & r_5 & r_6 & t_y
# \end{bmatrix}
# \begin{bmatrix}X\\Y\\Z\\1\end{bmatrix}
# $$
# Which leads, with further rearrangement, to an $Ab=0$ homogeneous system of equations to find $R,t$:
# $$
# \begin{bmatrix}
# \cdots \\
# X_i & Y_i & Z_i & 1 & 0    & 0    & 0    & 0  & -x'_iX_i & -x'_iY_i & -x'_iZ_i & -x'_i  \\
# 0    & 0    & 0    & 0  & X_i & Y_i & Z_i & 1 & -y'_iX_i & -y'_iY_i & -y'_iZ_i & -y'_i \\
# \cdots 
# \end{bmatrix}
# \begin{bmatrix}
# r_1 \\ r_2 \\ r_3 \\ t_x \\ r_4 \\ r_5 \\ r_6 \\ t_y \\ r_7 \\ r_8 \\ r_9 \\ t_z
# \end{bmatrix}
# = 0
# $$
# Which we can solve by solving the "minimal direction problem" ($\hat{b} = \mathop{\arg\min}_b|Ab|^2 \,\,\,\,\, \mathrm{s.t.} \,\,\, |b|=1
# $), which essentially means we take the SVD: $A=U\Sigma V^\top$, and take as the solution the last row of $V^\top$, and (according to this formulation) simply reshape it $3\times4$ to obtain P.
#
# However, the calculated matrix $P$ can take on an arbitrary scale, so the $R_{3\times3}$ matrix may need some conditioning to become a true rotation (orthonormal), effectively removing the scaling factor. Therefore we take the SVD and omit the scaling matrix $\Sigma$:
# $$
# \begin{align}
# R &= U\Sigma V^\top\\
# \hat{R} &= UV^\top
# \end{align}
# $$
# When we do that we need to impose the same rescaling on the translation: $\hat{t} = t\cdot\sum \hat{R}/R$

# %%
# write a function that takes in a set of 2D-3D correspondences and returns the camera pose
# this function should use the DLT (direct linear transform) algorithm we wrote above
# populate the A matrix with 2 entries from each correspondence according to the equations above
# then use SVD to solve for the camera pose P (3x4 matrix). 
# the pose would be the last column of the V matrix from the decomposition of A (i.e. V[:,-1]).
# if the Rotation component (left 3x3 submatrix) of the 3x4 pose matrix has a negative determinant, 
# multiply it by -1
# apply the method above to make sure the rotation is orthonormal. take its SVD and reassemble
# the rotation matrix from the U and V matrices (e.g. R_conditioned = U @ V^T)
# scale the translation (the 3x1 right submatrix of P) by the sum (R_conditioned / R)
# return R_conditioned, t_conditioned
def calculateCameraPose(corresp2D3D_):
    # ...
    return R,t


# %% [markdown]
# Again we use a "RANSAC" type method to find a robust solution while weeding out the outliers. We take a lax margin for inlier inclusion.

# %%
# get the 2D-3D correspondences from the graph (as well as "backmapping" so we know how to map back 
# to the originating 2D points).
correspond2D3D, backmapping = mm.aligned2D3D(5)
# the correspond2D3D array is a 3D array of shape (N,2,3) where N is the number of 2D-3D correspondences
# the first index is the correspondence pair index
# the second index is 0 for the 2D point (homogeneous, so it's 1x3) and 1 for the 3D point (1x3)
# for example correspond2D3D[0,0] is the 2D point and correspond2D3D[0,1] is the 3D point

# here's another opportunity to use the RANSAC algorithm to find the best camera pose
# use the function you wrote above to calculate the camera pose for a random sample of 6 correspondences
# then find the inliers (2D points that are within 25 pixels of the reprojected 3D points)
# using cv2.projectPoints, and np.linalg.norm (and .squeeze() as needed).
# the best camera pose is the one with the most inliers (use the maxv variable below to keep track of this)
# save the final camera pose in R_final and t_final
maxv = 0
for i in range(25000):
    # ...

# %%
# pritnt your results
R_final,t_final

# %%
# again compare your results to OpenCV's solvePnPRansac function
# you rsults should be very similar, but not exactly the same. a small difference is expected
# you very well can run the RANSAC loop above multiple times and get other reuslts
_,r_cv,t_cv,inliers_cv = cv2.solvePnPRansac(correspond2D3D[:,1], correspond2D3D[:,0,:2], K, None)
print(cv2.Rodrigues(r_cv)[0], t_cv.T, len(inliers_cv))

# %% [markdown]
# Make sure the process above finds > 30% of the points as inliers. If it doesn't - run it again, it's a game of chance.

# %%
# here's a visualization of the 2D-3D correspondences and the reprojected 3D points
# the red points are the 2D points and the blue points are the reprojected 3D points
# the reprojected 3D points should be close to the 2D points - but we gave it a good margin of 25 pixels
# so we expect some of the blue points to be outside the red points.
# outliars are points that are very far from the reprojected 3D points
projPts2d2,_ = cv2.projectPoints(correspond2D3D[:,1].T, R_final,t_final, K, None)

plt.figure(figsize=(20,10))
plt.imshow(images[5])
plt.scatter(correspond2D3D[:,0,0],correspond2D3D[:,0,1],label='2D Image', c='r')
plt.scatter(projPts2d2[:,0,0],projPts2d2[:,0,1],label='Reprojected 3D',c='b',s=5)
plt.legend();


# %% [markdown]
# That looks pretty bad doesn't it!
#
# We can do much better with the following step...
#
# Luckily (here is the power of non-linear optimization), if the initial solution is even remotely close (such as the case above), we can make pretty big steps towards a far better solution.

# %% [markdown]
# #### Camera Pose Non-Linear Optimization
#
# This time we complement with a non-linear least squares optimization, with a huber loss to help further with outliers, to minimize the reprojection loss: 
# $$
# \displaystyle
# \hat{P} = \mathop{\arg\min}_{P} \sum_i \Vert \mathrm{Proj}(P,X_i^{\mathrm{3D}}) - x_i^{\mathrm{2D}} \Vert
# $$

# %%
# write a function that calculates the residuals (error) between 2D and reprojected-3D (also 2D) points given 
# the rotation and translation vectors (flattened into a 1D array, see below)
# use the cv2.projectPoints function to reproject the 3D points (correspond2D3D[:,1]) to 2D
# return the residuals (just a subtraction `p2D - pReproj2D`) as a 1D array (use the ravel() function)
# the original 2D points are in correspond2D3D[:,0,:2]
def calcResiduals(Rt):
    # ...
    
# use scipy least squares to refine the pose estimate (scipy.optimize.least_squares)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
# however this time we will use a robus method (Huber loss function, e.g. loss='huber') to deal with outliers, as well as use
# an iterative solver which works with the residuals.
# the parameters to optimize are the rotation vector and the translation vector - in rodrigues form (cv2.Rodrigues)
# stack them together (hstack) and use the ravel() function to convert to a 1D array, this will be the input to the
# calcResiduals function you defined above.

# res = scipy.optimize.least_squares(...


# %%
# extract the solution from the least squares function's result
R2 = cv2.Rodrigues(res.x[:3])[0]
t2 = res.x[3:]

# %%
# our optimized pose
R2, t2

# %%
# compare again to the OpenCV solution (r_cv, t_cv) and see that now they are virtually the same
# the difference should now be very very small.
cv2.Rodrigues(r_cv)[0], t_cv.T

# %% [markdown]
# Display the new projected points to verify the solution

# %%
# let's visualize the results again to see how well the reprojected 3D points match the 2D points
projPts2d2,_ = cv2.projectPoints(correspond2D3D[:,1].T, R2, t2, K, None)

plt.figure(figsize=(20,10))
plt.imshow(mm.images[5])
plt.scatter(correspond2D3D[:,0,0],correspond2D3D[:,0,1],label='2D Image', c='r', s=40)
plt.scatter(projPts2d2[:,0,0],projPts2d2[:,0,1],label='Reprojected 3D', c='b', s=20)
plt.legend();

# %% [markdown]
# Now that looks much better!

# %%
# this is code provided to you to add the camera pose to the map.

# add the camera pose to the map
mm.poses[5] = np.hstack([R2,t2[np.newaxis].T])

# add this view to the visibility graph match_graph_3d[5], where the reprojection 
# error is small (3d points we are confident are viewable from camera 5)
projPts2d2,_ = cv2.projectPoints(correspond2D3D[:,1].T, mm.R(5), mm.t(5), K, None)
for i in range(len(projPts2d2)):
    if np.linalg.norm(np.squeeze(projPts2d2[i]) - correspond2D3D[i,0,:2]) < 5:
        mm.point3d_camera_visibility[5, backmapping[i][0]] = backmapping[i][1]

# %% [markdown]
# Let's have a look at the 3D point visibility map. Now we've added camera 5, it overlaps with some of the points, but some were thrown out because they're outliers.

# %%
plt.figure(figsize=(20,10))
plt.imshow((mm.point3d_camera_visibility > 0)[:,:100])

# %% [markdown]
# ### Triangulate more points
#
# Now that we found the pose for camera 5, let's add more 3D points to the reconstruction that come from the pairs: 3-5, and 5-7 (but don't already exist in the current 3-5-7 map)

# %%
# let's get aligned 2D points from 3,5 views that are not in the map
# the backidxL and backidxR are the indices of the 2D points in the original keypoint arrays in the map
pt2dl, pt2dr, backidxL, backidxR = mm.aligned2DNotInMap(3, 5)

# %%
# now use cv2.triangulatePoints to triangulate the 2D points.
# you will need to convert the 2D points to normalized coordinates before triangulation
# that is, multiply on the left by the inverse of the camera matrix Kinv: 
#   np.matmul(Kinv, cv2.convertPointsToHomogeneous(pt2d).squeeze().T).T
# the output of cv2.triangulatePoints is in homogeneous coordinates, so you will need to convert it to
# euclidean coordinates using cv2.convertPointsFromHomogeneous.
# make sure your final output is a 3xN array (N is the number of points)
pts3d = ...

# %%
# here is some code to visualize the results
# the blue points are the reprojected 3D points triangulated from the 2D
# the red points are the original image 2D points
projPts2d2,_ = cv2.projectPoints(pts3d, mm.R(5), mm.t(5), K, None)

plt.figure(figsize=(20,10))
plt.imshow(mm.images[5])
plt.scatter(pt2dr[:,0],pt2dr[:,1],label='2D Image', c='r', s=40)
plt.scatter(projPts2d2[:,0,0],projPts2d2[:,0,1],label='Reprojected 3D', c='b', s=20)
plt.legend();

# %%
# append the new 3D points to the map
M, N = mm.map_3d.shape[0], pts3d.shape[0]
mm.point3d_camera_visibility[3, M:M+N] = backidxL
mm.point3d_camera_visibility[5, M:M+N] = backidxR
mm.map_3d = np.vstack([mm.map_3d, pts3d.squeeze()])

# %%
# visualize the map
plt.scatter(mm.map_3d[:,0],mm.map_3d[:,2])
plt.xlim(-1,1)
plt.ylim(0.5,2.5)
plt.xlabel('X')
plt.ylabel('Z')
plt.title('X-Z plane (view from above)');

# %%
# and the visibility graph
plt.figure(figsize=(20,10))
plt.imshow((mm.point3d_camera_visibility > 0)[:,:400])

# %% [markdown]
# ---
# ## 3. Bundle Adjustment
#
# Write a bundle adjuster to optimize the reconstruction in Q.2: Find the optimal camera intrinsics, extrinsics (camera pose) and 3D points, such that the reprojection error (where there is visibility, guarded by $w_{ij}$) is minimal:
# $$
# \hat{X}, \hat{C}, \hat{K} = \mathop{\arg\min}_{X,C,K} \sum_j^M \sum_i^N w_{ij} \left\Vert \mathrm{Proj}(X_i^\mathrm{(3D)},C_j,K) - x_i^\mathrm{(2D)} \right\Vert^2
# $$
#
# Assume:
# 1. All cameras have the same K matrix
# 2. Pixels are square ($f_x = f_y$)
# 3. There is no skew ($K_{0,1} = 0$)
#
# Pack all the parameters for this reconstruction into a single (very long) vector like so:
# $$
# \left[R^0_{1\times3}, R^1_{1\times3}, R^2_{1\times3}, t^0_{1\times3}, t^1_{1\times3}, t^2_{1\times3}, f, c_x, c_y, p^{0}_{\mathrm{3D}},\dots,p^{N}_{\mathrm{3D}}\right]
# $$
# Rotation matrices $3\times3$ should be converted to Rodrigues formula $1\times3$ (`cv2.Rodrigues(...)[0]`).

# %%
map_3d, mpts2DForViews, visibility = mm.alignedMapTo2DAndVisibility()

# %%
# create the vector of parameters for initialization
# the number of parameters is: 
#   3 for each camera rotation, 
#   3 for each camera translation, 
#   3 for the intrinsics
#   and 3 for each 3D point
n_cams = 3
cam_ids = [3,5,7]
n_pts = mm.map_3d.shape[0]
params_size = n_cams * 6 + n_pts * 3 + 3

# the params vector. you will need to fill this in with the correct values
x0 = np.zeros((params_size,), np.float32) 

# initialize the camera rotations and translations (6 params per cam) starting at 0
# intrinsics start n_cams * 6, 3 params total
# 3d points start at K_idx + 3, 3 params per point

# fill in the params vector
# for example, the first camera rotation (1x3 vector) is x0[0:3]. the j'th camera rotation is x0[j*3:j*3+3]
# get the rotation from mm.R(camId) and translation (1x3 again) from mm.t(camId)
# the intrinsics are K[0,0], K[0,2], K[1,2], and they go into x0[K_idx+0], x0[K_idx+1], x0[K_idx+2], 
# respectively where K_idx is the index of the first intrinsics parameter (e.g. after the cameras)
# finally add the 3D points to the params vector as well

# ... your code here ...

# %% [markdown]
# This function calculates the residuals vector, e.g. $\left[\dots,r_{ji}^x,r_{ji}^y,\dots\right]$ where $r_{ji} = \left(\mathrm{Proj}(P^{\mathrm{3D}}_i,C_j,K)-p^{\mathrm{2D}}_i\right)$, unpacking the parameters for $P^{\mathrm{3D}}_i,C_j,K$ from the `params` vector, and taking the 2D point $p^{\mathrm{2D}}_i$ from the `pts2d` argument. The shape of the residuals vector is $1\times2N$, where $N$ is the number of points. 

# %%
# this function will be called by the optimizer to calculate the reprojection error (residuals)
# the "params" vector contains the camera poses and 3D points flattened into a single vector
# the "n_cams" and "n_pts" variables are the number of cameras and 3D points in the map
# the "cam_ids" are camera ids (e.g. [3,5,7]), so we can iterate [0]->3, [1]->5, [2]->7
# the "pts2d" are image 2D points that we want to compare to the reprojection (some of them are "empty")
# the "visibility" is a matrix that indicates which 3D points are visible from which camera
# the optimization cannot change "pts2d" and "visibility", these are our "truth"
# the "show_debug" variable is used to visualize the reprojection error
# the function returns a residual vector of the reprojection error (`a - b`) for each 2D point
def calcResidualsBA(params, n_cams, n_pts, cam_ids, pts2d, visibility, show_debug=False):
    # unpack the camera poses and 3D points from the "params" vector
    # params = [R1 (1x3), t1 (1x3), ..., Rn (1x3), tn (1x3), f, cx, cy, X1, Y1, Z1, ..., Xn, Yn, Zn]
    # a total of 6*n_cams + 3 + 3*n_pts parameters
    # the camera poses are stored as Rodrigues vectors (3 parameters) and translations (3 parameters)
    # the intrinsics are stored as focal length (1 parameter), and 2 principal point parameters
    # the 3D points are stored as X,Y,Z coordinates (3 parameters each)
    # extract Rs, ts, f, cx, cy, pts3d from the vector using the scheme above
    # build intrinsics K from f, cx, cy
    # then, for each camera (Rs[i],ts[i]) project the 3D points into the image plane with cv2.projectPoints
    # for every projected point - check if it's visible in the image (visibility[cam_ids[i]][j] == 1)
    # if it's visible, calculate the reprojection error (a - b) and append it to the "resid" vector
    # if it's not visible, append [0,0] to the "resid" vector
    # finally, return the "resid" vector stacked into a single vector (np.hstack)

    # ... your code here ...


# %% [markdown]
# This builds the sparsity matrix of the jacobian (partial derivative matrix), which greatly increases the speed of optimization. We let the optimizer calculate the jacobian by itself numerically (by adding small $\Delta$s to the parameters), which is a costly operation, therefore we guide it by saying what elements of the jacobian matrix will always be 0 and never have to be calculated. 
#
# For each 2D point residual, many parameters for optimization in the system are irrelevant, and in fact a 2D residual is derived from just a handful of parameters: $K,C_j,P^{\mathrm{3D}}_i$, therefore the jacobian is very sparse. 

# %%
from scipy.sparse import lil_matrix

# this function creates a sparse matrix that indicates which parameters affect which residuals
# this is used by the optimizer to speed up the calculation
# the "n_cams" and "n_pts" variables are the number of cameras and 3D points in the map
# the "cam_ids" are camera ids (e.g. [3,5,7]), so we can iterate [0]->3, [1]->5, [2]->7
# the "visibility" is a matrix that indicates which 3D points are visible from which camera,
#   this is used to determine which residuals are affected by which parameters.
# total number of potential residuals (some will be zero) = n_cams * n_pts * 2
# total number of parameters = n_cams * 6 + n_pts * 3 + 3

# so build a sparse matrix (`lil_matrix`) A of size (n_cams * n_pts * 2) x (n_cams * 6 + n_pts * 3 + 3)
# in the residual vector, the camera parameters take up the first 6*n_cams parameters
# the intrinsic parameters take up the next 3 parameters, starting at 6*n_cams
# the 3D points take up the next 3*n_pts parameters, starting at 6*n_cams + 3.
# to figure out where in the residual vector each parameter affects, use the following scheme:
#   each residual (reprojection) is effected by the camera pose (R,t) that it was projected from
#   each residual is also effected by the 3D point that was projected
#   each residual is also effected by the intrinsic parameters (f, cx, cy)
# the poses, 3d points and intrinsics are stored in the "params" vector in the order that we've
#   described above, so you can use the same indexing scheme to figure out where each parameter.
# 
# for example, a point `i`` in camera `j`` will affect sparsity matrix at 
#   rows: `2*i + j*n_pts*2` and `2*i + j*n_pts*2 + 1`
#   columns: `j*3` -> `j*3 + 3` (the 3 parameters of the camera rotation)
#            `ts_ids + j*3` -> `ts_ids + j*3 + 3` (the 3 parameters of the camera translation)
#            `K_idx` -> `K_idx + 3` (the 3 parameters of the intrinsics)
#            `pts3d_idx + i*3` -> `pts3d_idx + i*3 + 3` (the 3 parameters of the 3D point)
#
# sparse non-linear optimization problems are not easy to set up. so pay attention to the details.
def bundle_adjustment_sparsity(n_cams, n_pts, cam_ids, visibility):
    # ... your code here ...


# %%
A = bundle_adjustment_sparsity(3, map_3d.shape[0], [3, 5, 7], visibility)

# %% [markdown]
# To illustrate the sparsity of the jacobian:

# %%
plt.spy(A,markersize=0.03);

# %% [markdown]
# Some parameters for optimization should be bound, and not be allowed to get extreme values. For example the focal length $f$ as well as $c_x,c_y$ cannot be negative, and $f$ should be capped above at e.g. 2000.

# %%
# set up bounds on the K (intrinsics parameters)
bounds = (np.full((params_size,),-np.inf),np.full((params_size,),np.inf))
bounds[0][K_idx:K_idx+3] = [100,0,0]
bounds[1][K_idx:K_idx+3] = [800,mm.images[0].shape[1],mm.images[0].shape[0]]

# %% [markdown]
# Run the NLLSQ optimizer:

# %%
res = scipy.optimize.least_squares(calcResidualsBA, x0, 
                                   jac_sparsity=A, 
                                   verbose=2, 
                                   x_scale='jac', 
                                   ftol=1e-5, 
                                   jac='3-point',
                                   bounds=bounds,
                                   args=(n_cams, n_pts, [3, 5, 7], mpts2DForViews, visibility, False))

# %% [markdown]
# Lets have a look on the effect the optimization had on the residuals:

# %%
plt.plot(calcResidualsBA(x0, n_cams, n_pts, [3, 5, 7], mpts2DForViews, visibility, False),label='original')
plt.plot(calcResidualsBA(res.x, n_cams, n_pts, [3, 5, 7], mpts2DForViews, visibility, False),label='optimized')
plt.title('Residuals')
plt.legend(fontsize='xx-large');

# %% [markdown]
# Seems the residuals have mostly all improved.
#
# Now visually on the images with the points projections:

# %%
plt.figure(figsize=(20,5))
calcResidualsBA(res.x, n_cams, n_pts, [3, 5, 7], mpts2DForViews, visibility, True);

# %% [markdown]
# This looks much better than before, right?
#
# Collect the optimized measurements from the parameters vector:

# %%
# unpack the results vector.
# save in the variables `pts3d_hat`, `ts_hat`, `Rs_hat`, `K_hat`
# the location of these parameters in the results vector is the same as in the sparsity matrix and as before.
# ... your code here ...

# %% [markdown]
# Show the 3D points (top-view, XZ plane):

# %%
# visualize the results to see if they make sense
plt.figure(figsize=(10,10))
plt.scatter(pts3d_hat[:,0],pts3d_hat[:,2], label='3D points')
for cam_i in range(n_cams):
    Ri = cv2.Rodrigues(Rs_hat[cam_i])[0]
    ti = ts_hat[cam_i]
    ti = -Ri.T @ ti
    Ri = Ri.T
    plt.scatter(ti[0],ti[2],s=100, label='Cameras')
    plt.quiver(ti[0],ti[2],Ri[0,2],Ri[0,0],label='Direction')

plt.xlabel('X', fontsize='xx-large'),plt.ylabel('Z', fontsize='xx-large')
plt.xlim(-1,1),plt.ylim(-0.1,1.9)
plt.legend()

# %% [markdown]
# ---

# %% [markdown]
# That's a wrap!
#
# You have built a 3D reconstrution pipeline from scratch. This is a big deal!
#
# You can use this technique and extend it to work with images of your own, and make dense reconstructions with stereo matching (functions for which exist in OpenCV).
#
# We will see later the evolution of these methods in the world of deep learning and big datasets.

# %%
