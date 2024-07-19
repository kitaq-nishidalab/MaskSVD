from torch.utils.data import Dataset
import torch
import os
import glob
import h5py
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from torch import sin, cos
import open3d as o3d
import random

def download_modelnet40():
	BASE_DIR = "/home/nishidalab0/MaskNet/learning3d"
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	if not os.path.exists(DATA_DIR):
		os.mkdir(DATA_DIR)
	if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
		www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
		zipfile = os.path.basename(www)
		os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
		os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
		os.system('rm %s' % (zipfile))
		
def load_data(train, use_normals):
	if train: partition = 'train'
	else: partition = 'test'
	BASE_DIR = "/home/nishidalab0/MaskNet/learning3d"
	DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	all_data = []
	all_label = []
	for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
		f = h5py.File(h5_name)
		if use_normals: data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1).astype('float32')
		else: data = f['data'][:].astype('float32')
		label = f['label'][:].astype('int64')
		f.close()
		all_data.append(data)
		all_label.append(label)
	all_data = np.concatenate(all_data, axis=0)
	all_label = np.concatenate(all_label, axis=0)
	return all_data, all_label
	
class ModelNet40Data(Dataset):
	def __init__(
		self,
		train=True,
		num_points=1024,
		download=True,
		randomize_data=False,
		use_normals=False
	):
		super(ModelNet40Data, self).__init__()
		if download: download_modelnet40()
		self.data, self.labels = load_data(train, use_normals)
		if not train: self.shapes = self.read_classes_ModelNet40()
		self.num_points = num_points
		self.randomize_data = randomize_data

	def __getitem__(self, idx):
		if self.randomize_data: current_points = self.randomize(idx)
		else: current_points = self.data[idx].copy()

		current_points = torch.from_numpy(current_points[:self.num_points, :]).float()
		label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)

		return current_points, label

	def __len__(self):
		return self.data.shape[0]

	def randomize(self, idx):
		pt_idxs = np.arange(0, self.num_points)
		np.random.shuffle(pt_idxs)
		return self.data[idx, pt_idxs].copy()

	def get_shape(self, label):
		return self.shapes[label]
		
	def read_classes_ModelNet40(self):
		BASE_DIR = "/home/nishidalab0/MaskNet/learning3d"
		DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
		file = open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r')
		shape_names = file.read()
		shape_names = np.array(shape_names.split('\n')[:-1])
		return shape_names
		
###########################
#sinc
###########################

def sinc1(t):
    """ sinc1: t -> sin(t)/t """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t[s] ** 2
    r[s] = 1 - t2/6*(1 - t2/20*(1 - t2/42))  # Taylor series O(t^8)
    r[c] = sin(t[c]) / t[c]

    return r

def sinc2(t):
    """ sinc2: t -> (1 - cos(t)) / (t**2) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t ** 2
    r[s] = 1/2*(1-t2[s]/12*(1-t2[s]/30*(1-t2[s]/56)))  # Taylor series O(t^8)
    r[c] = (1-cos(t[c]))/t2[c]

    return r

def sinc3(t):
    """ sinc3: t -> (t - sin(t)) / (t**3) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t[s] ** 2
    r[s] = 1/6*(1-t2/20*(1-t2/42*(1-t2/72)))  # Taylor series O(t^8)
    r[c] = (t[c]-sin(t[c]))/(t[c]**3)

    return r

###########################
#so3
###########################
def mat(x):
    # size: [*, 3] -> [*, 3, 3]
    x_ = x.view(-1, 3)
    x1, x2, x3 = x_[:, 0], x_[:, 1], x_[:, 2]
    O = torch.zeros_like(x1)

    X = torch.stack((
        torch.stack((O, -x3, x2), dim=1),
        torch.stack((x3, O, -x1), dim=1),
        torch.stack((-x2, x1, O), dim=1)), dim=1)
    return X.view(*(x.size()[0:-1]), 3, 3)
    
###########################
#se3
###########################
def exp(x):
    x_ = x.view(-1, 6)
    w, v = x_[:, 0:3], x_[:, 3:6]
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    W = mat(w)
    S = W.bmm(W)
    I = torch.eye(3).to(w)

    # Rodrigues' rotation formula.
    #R = cos(t)*eye(3) + sinc1(t)*W + sinc2(t)*(w*w');
    #  = eye(3) + sinc1(t)*W + sinc2(t)*S
    R = I + sinc1(t)*W + sinc2(t)*S

    #V = sinc1(t)*eye(3) + sinc2(t)*W + sinc3(t)*(w*w')
    #  = eye(3) + sinc2(t)*W + sinc3(t)*S
    V = I + sinc2(t)*W + sinc3(t)*S

    p = V.bmm(v.contiguous().view(-1, 3, 1))

    z = torch.Tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(x_.size(0), 1, 1).to(x)
    Rp = torch.cat((R, p), dim=2)
    g = torch.cat((Rp, z), dim=1)

    return g.view(*(x.size()[0:-1]), 4, 4)


def transform(g, a):
    # g : SE(3),  * x 4 x 4
    # a : R^3,    * x 3[x N]
    g_ = g.view(-1, 4, 4)
    R = g_[:, 0:3, 0:3].contiguous().view(*(g.size()[0:-2]), 3, 3)
    p = g_[:, 0:3, 3].contiguous().view(*(g.size()[0:-2]), 3)
    if len(g.size()) == len(a.size()):
        b = R.matmul(a) + p.unsqueeze(-1)
    else:
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p
    return b
    
############################
#quaternion
#############################

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == "xyz":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == "yzx":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == "zxy":
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "xzy":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == "yxz":
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "zyx":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)


# Numpy-backed implementations


def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()


def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return qrot(q, v).numpy()


def qeuler_np(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return qeuler(q, order, epsilon).numpy()


def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack(
        (np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1
    )
    ry = np.stack(
        (np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1
    )
    rz = np.stack(
        (np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1
    )

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1

    return result.reshape(original_shape)


def qinv(q):
    # expectes q in (w,x,y,z) format
    w = q[:, 0:1]
    v = q[:, 1:]
    inv = torch.cat([w, -v], dim=1)
    return inv

##############
#registration
###############

def knn_idx(pts, k):
	kdt = cKDTree(pts)
	_, idx = kdt.query(pts, k=k+1)
	return idx[:, 1:]
	
def get_rri(pts, k):
	# pts: N x 3, original points
	# q: N x K x 3, nearest neighbors
	q = pts[knn_idx(pts, k)]
	p = np.repeat(pts[:, None], k, axis=1)
	# rp, rq: N x K x 1, norms
	rp = np.linalg.norm(p, axis=-1, keepdims=True)
	rq = np.linalg.norm(q, axis=-1, keepdims=True)
	pn = p / rp
	qn = q / rq
	dot = np.sum(pn * qn, -1, keepdims=True)
	# theta: N x K x 1, angles
	theta = np.arccos(np.clip(dot, -1, 1))
	T_q = q - dot * p
	sin_psi = np.sum(np.cross(T_q[:, None], T_q[:, :, None]) * pn[:, None], -1)
	cos_psi = np.sum(T_q[:, None] * T_q[:, :, None], -1)
	psi = np.arctan2(sin_psi, cos_psi) % (2*np.pi)
	idx = np.argpartition(psi, 1)[:, :, 1:2]
	# phi: N x K x 1, projection angles
	phi = np.take_along_axis(psi, idx, axis=-1)
	feat = np.concatenate([rp, rq, theta, phi], axis=-1)
	return feat.reshape(-1, k * 4)

def get_rri_cuda(pts, k, npts_per_block=1):
	try:
		import pycuda.autoinit
		from pycuda import gpuarray
		from pycuda.compiler import SourceModule
	except Exception as e:
		print("Error raised in pycuda modules! pycuda only works with GPU, ", e)
		raise

def jitter_pointcloud(pointcloud, sigma=0.04, clip=0.05):
	# N, C = pointcloud.shape
	sigma = 0.04*np.random.random_sample()
	pointcloud += torch.empty(pointcloud.shape).normal_(mean=0, std=sigma).clamp(-clip, clip)
	return pointcloud

#切り分けのときに使うやつ	
def farthest_subsample_points(pointcloud1, num_subsampled_points=102):
	pointcloud1 = pointcloud1
	num_points = pointcloud1.shape[0]
	nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
							 metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
	random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
	idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
	gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)
	return pointcloud1[idx1, :], gt_mask

def plane_subsample_points(pointcloud, num_subsampled_points=100):
    pointcloud = pointcloud
    num_points = pointcloud.shape[0]
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pointcloud.to('cpu').detach().numpy().copy())

    plane_all, inliers_all = pcd_o3d.segment_plane(distance_threshold=0.1,
                                                      ransac_n=3,
                                                      num_iterations=1000)
    if len(inliers_all) >= num_subsampled_points:
        # 平面上の点群をサブサンプリング
        combined_indices = random.sample(inliers_all, min(num_subsampled_points, len(inliers_all)))
    else:
        # 平面上の点のインデックス以外の点を選択
        outliers_all = [i for i in range(num_points) if i not in inliers_all]
        # 平面外の点からランダムにサンプリング
        num_outliers_to_sample = num_subsampled_points - len(inliers_all)
        random_indices = random.sample(outliers_all, num_outliers_to_sample)
        # random_indicesとinliers_allを連結
        combined_indices = random_indices + inliers_all
        #print(combined_indices)
    
    gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(combined_indices), 1)
    return pointcloud[combined_indices, :], gt_mask
	
class PNLKTransform:
    """ rigid motion """
    def __init__(self, mag=1, mag_randomly=False):
        self.mag = mag
        self.randomly = mag_randomly

        self.gt = None
        self.igt = None
        self.index = 0

    def generate_transform(self):
        # return: a twist-vector
        amp = self.mag
        if self.randomly:
            amp = torch.rand(1, 1) * self.mag
        x = torch.randn(1, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp

        return x # [1, 6]

    def apply_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        g = exp(x).to(p0)   # [1, 4, 4]
        gt = exp(-x).to(p0) # [1, 4, 4]

        p1 = transform(g, p0)
        self.gt = gt.squeeze(0) #  gt: p1 -> p0
        self.igt = g.squeeze(0) # igt: p0 -> p1
        return p1

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)


class RPMNetTransform:
    """ rigid motion """
    def __init__(self, mag=1, mag_randomly=False):
        self.mag = mag
        self.randomly = mag_randomly

        self.gt = None
        self.igt = None
        self.index = 0

    def generate_transform(self):
        # return: a twist-vector
        amp = self.mag
        if self.randomly:
            amp = torch.rand(1, 1) * self.mag
        x = torch.randn(1, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp

        return x # [1, 6]

    def apply_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        g = exp(x).to(p0)   # [1, 4, 4]
        gt = exp(-x).to(p0) # [1, 4, 4]

        p1 = transform(g, p0[:, :3])

        if p0.shape[1] == 6:  # Need to rotate normals also
            g_n = g.clone()
            g_n[:, :3, 3] = 0.0
            n1 = transform(g_n, p0[:, 3:6])
            p1 = torch.cat([p1, n1], axis=-1)

        self.gt = gt.squeeze(0) #  gt: p1 -> p0
        self.igt = g.squeeze(0) # igt: p0 -> p1
        return p1

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)


class PCRNetTransform:
    def __init__(self, data_size, angle_range=45, translation_range=1):
        self.angle_range = angle_range
        self.translation_range = translation_range
        self.dtype = torch.float32
        self.transformations = [self.create_random_transform(torch.float32, self.angle_range, self.translation_range) for _ in range(data_size)]
        self.index = 0

    @staticmethod
    def deg_to_rad(deg):
        return np.pi / 180 * deg

    def create_random_transform(self, dtype, max_rotation_deg, max_translation):
        max_rotation = self.deg_to_rad(max_rotation_deg)
        rot = np.random.uniform(-max_rotation, max_rotation, [1, 3])
        trans = np.random.uniform(-max_translation, max_translation, [1, 3])
        quat = euler_to_quaternion(rot, "xyz")

        vec = np.concatenate([quat, trans], axis=1)
        vec = torch.tensor(vec, dtype=dtype)
        return vec

    @staticmethod
    def create_pose_7d(vector: torch.Tensor):
        # Normalize the quaternion.
        pre_normalized_quaternion = vector[:, 0:4]
        normalized_quaternion = F.normalize(pre_normalized_quaternion, dim=1)

        # B x 7 vector of 4 quaternions and 3 translation parameters
        translation = vector[:, 4:]
        vector = torch.cat([normalized_quaternion, translation], dim=1)
        return vector.view([-1, 7])

    @staticmethod
    def get_quaternion(pose_7d: torch.Tensor):
        return pose_7d[:, 0:4]

    @staticmethod
    def get_translation(pose_7d: torch.Tensor):
        return pose_7d[:, 4:]

    @staticmethod
    def quaternion_rotate(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        ndim = point_cloud.dim()
        if ndim == 2:
            N, _ = point_cloud.shape
            assert pose_7d.shape[0] == 1
            # repeat transformation vector for each point in shape
            quat = PCRNetTransform.get_quaternion(pose_7d).expand([N, -1])
            rotated_point_cloud = qrot(quat, point_cloud)

        elif ndim == 3:
            B, N, _ = point_cloud.shape
            quat = PCRNetTransform.get_quaternion(pose_7d).unsqueeze(1).expand([-1, N, -1]).contiguous()
            rotated_point_cloud = qrot(quat, point_cloud)

        return rotated_point_cloud

    @staticmethod
    def quaternion_transform(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        transformed_point_cloud = PCRNetTransform.quaternion_rotate(point_cloud, pose_7d) + PCRNetTransform.get_translation(pose_7d).view(-1, 1, 3).repeat(1, point_cloud.shape[1], 1)      # Ps' = R*Ps + t
        return transformed_point_cloud

    @staticmethod
    def convert2transformation(rotation_matrix: torch.Tensor, translation_vector: torch.Tensor):
        one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(rotation_matrix.shape[0], 1, 1).to(rotation_matrix)    # (Bx1x4)
        transformation_matrix = torch.cat([rotation_matrix, translation_vector[:,0,:].unsqueeze(-1)], dim=2)                        # (Bx3x4)
        transformation_matrix = torch.cat([transformation_matrix, one_], dim=1)                                     # (Bx4x4)
        return transformation_matrix

    def __call__(self, template):
        self.igt = self.transformations[self.index]
        gt = self.create_pose_7d(self.igt)
        source = self.quaternion_rotate(template, gt) + self.get_translation(gt)
        return source


class DCPTransform:
    def __init__(self, angle_range=45, translation_range=1):
        self.angle_range = angle_range*(np.pi/180)
        self.translation_range = translation_range
        self.index = 0

    def generate_transform(self):
        self.anglex = np.random.uniform() * self.angle_range
        self.angley = np.random.uniform() * self.angle_range
        self.anglez = np.random.uniform() * self.angle_range
        self.translation = np.array([np.random.uniform(-self.translation_range, self.translation_range),
                                        np.random.uniform(-self.translation_range, self.translation_range),
                                        np.random.uniform(-self.translation_range, self.translation_range)])
        # cosx = np.cos(self.anglex)
        # cosy = np.cos(self.angley)
        # cosz = np.cos(self.anglez)
        # sinx = np.sin(self.anglex)
        # siny = np.sin(self.angley)
        # sinz = np.sin(self.anglez)
        # Rx = np.array([[1, 0, 0],
        #                 [0, cosx, -sinx],
        #                 [0, sinx, cosx]])
        # Ry = np.array([[cosy, 0, siny],
        #                 [0, 1, 0],
        #                 [-siny, 0, cosy]])
        # Rz = np.array([[cosz, -sinz, 0],
        #                 [sinz, cosz, 0],
        #                 [0, 0, 1]])
        # self.R_ab = Rx.dot(Ry).dot(Rz)
        # last_row = np.array([[0., 0., 0., 1.]])
        # self.igt = np.concatenate([self.R_ab, self.translation_ab.reshape(-1,1)], axis=1)
        # self.igt = np.concatenate([self.igt, last_row], axis=0)

    def apply_transformation(self, template):
        rotation = Rotation.from_euler('zyx', [self.anglez, self.angley, self.anglex])
        self.igt = rotation.apply(np.eye(3))
        self.igt = np.concatenate([self.igt, self.translation.reshape(-1,1)], axis=1)
        self.igt = torch.from_numpy(np.concatenate([self.igt, np.array([[0., 0., 0., 1.]])], axis=0)).float()
        source = rotation.apply(template) + np.expand_dims(self.translation, axis=0)
        return source

    def __call__(self, template):
        template = template.numpy()
        self.generate_transform()
        return torch.from_numpy(self.apply_transformation(template)).float()

class DeepGMRTransform:
    def __init__(self, angle_range=45, translation_range=1):
        self.angle_range = angle_range*(np.pi/180)
        self.translation_range = translation_range
        self.index = 0

    def generate_transform(self):
        self.anglex = np.random.uniform() * self.angle_range
        self.angley = np.random.uniform() * self.angle_range
        self.anglez = np.random.uniform() * self.angle_range
        self.translation = np.array([np.random.uniform(-self.translation_range, self.translation_range),
                                        np.random.uniform(-self.translation_range, self.translation_range),
                                        np.random.uniform(-self.translation_range, self.translation_range)])

    def apply_transformation(self, template):
        rotation = Rotation.from_euler('zyx', [self.anglez, self.angley, self.anglex])
        self.igt = rotation.apply(np.eye(3))
        self.igt = np.concatenate([self.igt, self.translation.reshape(-1,1)], axis=1)
        self.igt = torch.from_numpy(np.concatenate([self.igt, np.array([[0., 0., 0., 1.]])], axis=0)).float()
        source = rotation.apply(template) + np.expand_dims(self.translation, axis=0)
        return source

    def __call__(self, template):
        template = template.numpy()
        self.generate_transform()
        return torch.from_numpy(self.apply_transformation(template)).float()
        
class RegistrationData(Dataset):
	###def __init__(self, algorithm, data_class=ModelNet40Data(), partial_source=False, partial_template=False, noise=False, additional_params={'use_masknet': True}):
	def __init__(self, algorithm, data_class=ModelNet40Data(), partial_source=False, partial_template=False, noise=False, additional_params={'use_masknet': True}, num_subsampled_points=768):
		super(RegistrationData, self).__init__()
		available_algorithms = ['PCRNet', 'PointNetLK', 'DCP', 'PRNet', 'iPCRNet', 'RPMNet', 'DeepGMR']
		if algorithm in available_algorithms: self.algorithm = algorithm
		else: raise Exception("Algorithm not available for registration.")

		self.set_class(data_class)
		self.partial_template = partial_template
		self.partial_source = partial_source
		self.noise = noise
		self.additional_params = additional_params
		self.num_subsampled_points = num_subsampled_points
		self.use_rri = False

		if self.algorithm == 'PCRNet' or self.algorithm == 'iPCRNet':
			self.transforms = PCRNetTransform(len(data_class), angle_range=45, translation_range=1)
		if self.algorithm == 'PointNetLK':
			self.transforms = PNLKTransform(0.8, True)
		if self.algorithm == 'RPMNet':
			self.transforms = RPMNetTransform(0.8, True)
		if self.algorithm == 'DCP' or self.algorithm == 'PRNet':
			self.transforms = DCPTransform(angle_range=45, translation_range=1)
		if self.algorithm == 'DeepGMR':
			self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri
			self.transforms = DeepGMRTransform(angle_range=90, translation_range=1)
			if 'nearest_neighbors' in self.additional_params.keys() and self.additional_params['nearest_neighbors'] > 0:
				self.use_rri = True
				self.nearest_neighbors = self.additional_params['nearest_neighbors']

	def __len__(self):
		return len(self.data_class)

	def set_class(self, data_class):
		self.data_class = data_class

	def __getitem__(self, index):
		template, label = self.data_class[index]
		self.transforms.index = index				# for fixed transformations in PCRNet.
		source = self.transforms(template)

		# Check for Partial Data.
		###if self.partial_source: source, self.source_mask = farthest_subsample_points(source)
		###if self.partial_template: template, self.template_mask = farthest_subsample_points(template)
		# Check for Partial Data.(plane_subsample_points, farthest_subsample_points)
		if self.partial_source: source, self.source_mask = plane_subsample_points(source, num_subsampled_points=self.num_subsampled_points)
		if self.partial_template: template, self.template_mask = farthest_subsample_points(template, num_subsampled_points=self.num_subsampled_points)

		# Check for Noise in Source Data.
		if self.noise: source = jitter_pointcloud(source)

		if self.use_rri:
			template, source = template.numpy(), source.numpy()
			template = np.concatenate([template, self.get_rri(template - template.mean(axis=0), self.nearest_neighbors)], axis=1)
			source = np.concatenate([source, self.get_rri(source - source.mean(axis=0), self.nearest_neighbors)], axis=1)
			template, source = torch.tensor(template).float(), torch.tensor(source).float()

		igt = self.transforms.igt

		if self.additional_params['use_masknet']:
			if self.partial_source and self.partial_template:
				return template, source, igt, self.template_mask, self.source_mask
			elif self.partial_source:
				return template, source, igt, self.source_mask
			elif self.partial_template:
				return template, source, igt, self.template_mask
		else:
			return template, source, igt

