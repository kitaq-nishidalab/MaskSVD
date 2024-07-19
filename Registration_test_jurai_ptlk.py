import numpy as np
import torch

import open3d as o3d
import torch.nn as nn
import Global_optimizer
import os
from torch import sin, cos
import math
global theta
theta = 90


def rotation_matrix_to_euler_angles(R):
    # Extract angles using trigonometric relations
    roll = np.arctan2(-R[1, 2], R[2, 2])
    pitch = np.arctan2(-(R[0, 2]*np.cos(roll)), R[2, 2])
    yaw = np.arctan2(-R[0, 1], R[0, 0])

    return np.array([roll, pitch, yaw])

def coordinate_3d(axes, range_x, range_y, range_z, grid = True):
    axes.set_xlabel("x", fontsize = 14)
    axes.set_ylabel("y", fontsize = 14)
    axes.set_zlabel("z", fontsize = 14)
    axes.set_xlim(range_x[0], range_x[1])
    axes.set_ylim(range_y[0], range_y[1])
    axes.set_zlim(range_z[0], range_z[1])
    if grid == True:
        axes.grid()

def visual_vector_3d(axes, loc, vector, color = "red"):
    axes.quiver(loc[0], loc[1], loc[2],
              vector[0], vector[1], vector[2],
              color = color, lw=3)


###################
#Registration
###################
class PointNet(torch.nn.Module):
	def __init__(self, emb_dims=1024, input_shape="bnc", use_bn=False, global_feat=True):
		# emb_dims:			Embedding Dimensions for PointNet.
		# input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
		super(PointNet, self).__init__()
		if input_shape not in ["bcn", "bnc"]:
			raise ValueError("Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' ")
		self.input_shape = input_shape
		self.emb_dims = emb_dims
		self.use_bn = use_bn
		self.global_feat = global_feat
		if not self.global_feat: self.pooling = Pooling('max')

		self.layers = self.create_structure()

	def create_structure(self):
		self.conv1 = torch.nn.Conv1d(3, 64, 1)
		self.conv2 = torch.nn.Conv1d(64, 64, 1)
		self.conv3 = torch.nn.Conv1d(64, 64, 1)
		self.conv4 = torch.nn.Conv1d(64, 128, 1)
		self.conv5 = torch.nn.Conv1d(128, self.emb_dims, 1)
		self.relu = torch.nn.ReLU()

		if self.use_bn:
			self.bn1 = torch.nn.BatchNorm1d(64)
			self.bn2 = torch.nn.BatchNorm1d(64)
			self.bn3 = torch.nn.BatchNorm1d(64)
			self.bn4 = torch.nn.BatchNorm1d(128)
			self.bn5 = torch.nn.BatchNorm1d(self.emb_dims)

		if self.use_bn:
			layers = [self.conv1, self.bn1, self.relu,
					  self.conv2, self.bn2, self.relu,
					  self.conv3, self.bn3, self.relu,
					  self.conv4, self.bn4, self.relu,
					  self.conv5, self.bn5, self.relu]
		else:
			layers = [self.conv1, self.relu,
					  self.conv2, self.relu,
					  self.conv3, self.relu,
					  self.conv4, self.relu,
					  self.conv5, self.relu]
		return layers


	def forward(self, input_data):
		# input_data: 		Point Cloud having shape input_shape.
		# output:			PointNet features (Batch x emb_dims)
		if self.input_shape == "bnc":
			num_points = input_data.shape[1]
			input_data = input_data.permute(0, 2, 1)
		else:
			num_points = input_data.shape[2]
		if input_data.shape[1] != 3:
			raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

		output = input_data
		for idx, layer in enumerate(self.layers):
			output = layer(output)
			if idx == 1 and not self.global_feat: point_feature = output

		if self.global_feat:
			return output
		else:
			output = self.pooling(output)
			output = output.view(-1, self.emb_dims, 1).repeat(1, 1, num_points)
			return torch.cat([output, point_feature], 1)

class Pooling(torch.nn.Module):
	def __init__(self, pool_type='max'):
		self.pool_type = pool_type
		super(Pooling, self).__init__()

	def forward(self, input):
		if self.pool_type == 'max':
			return torch.max(input, 2)[0].contiguous()
		elif self.pool_type == 'avg' or self.pool_type == 'average':
			return torch.mean(input, 2).contiguous()


def batch_inverse(x):
    """ M(n) -> M(n); x -> x^-1 """
    batch_size, h, w = x.size()
    assert h == w
    y = torch.zeros_like(x)
    for i in range(batch_size):
        y[i, :, :] = x[i, :, :].inverse()
    return y

def batch_inverse_dx(y):
    """ backward """
    # Let y(x) = x^-1.
    # compute dy
    #   dy = dy(j,k)
    #      = - y(j,m) * dx(m,n) * y(n,k)
    #      = - y(j,m) * y(n,k) * dx(m,n)
    # therefore,
    #   dy(j,k)/dx(m,n) = - y(j,m) * y(n,k)
    batch_size, h, w = y.size()
    assert h == w
    # compute dy(j,k,m,n) = dy(j,k)/dx(m,n) = - y(j,m) * y(n,k)
    #   = - (y(j,:))' * y'(k,:)
    yl = y.repeat(1, 1, h).view(batch_size*h*h, h, 1)
    yr = y.transpose(1, 2).repeat(1, h, 1).view(batch_size*h*h, 1, h)
    dy = - yl.bmm(yr).view(batch_size, h, h, h, h)

    # compute dy(m,n,j,k) = dy(j,k)/dx(m,n) = - y(j,m) * y(n,k)
    #   = - (y'(m,:))' * y(n,:)
    #yl = y.transpose(1, 2).repeat(1, 1, h).view(batch_size*h*h, h, 1)
    #yr = y.repeat(1, h, 1).view(batch_size*h*h, 1, h)
    #dy = - yl.bmm(yr).view(batch_size, h, h, h, h)

    return dy

class InvMatrix(torch.autograd.Function):
    """ M(n) -> M(n); x -> x^-1.
    """
    @staticmethod
    def forward(ctx, x):
        y = batch_inverse(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors   # v0.4
        #y, = ctx.saved_variables  # v0.3.1
        batch_size, h, w = y.size()
        assert h == w

        # Let y(x) = x^-1 and assume any function f(y(x)).
        # compute df/dx(m,n)...
        #   df/dx(m,n) = df/dy(j,k) * dy(j,k)/dx(m,n)
        # well, df/dy is 'grad_output'
        # and so we will return 'grad_input = df/dy(j,k) * dy(j,k)/dx(m,n)'

        dy = batch_inverse_dx(y)  # dy(j,k,m,n) = dy(j,k)/dx(m,n)
        go = grad_output.contiguous().view(batch_size, 1, h*h)  # [1, (j*k)]
        ym = dy.view(batch_size, h*h, h*h)  # [(j*k), (m*n)]
        r = go.bmm(ym)  # [1, (m*n)]
        grad_input = r.view(batch_size, h, h)  # [m, n]

        return grad_input

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


def so3_mat(x):
    # size: [*, 3] -> [*, 3, 3]
    x_ = x.view(-1, 3)
    x1, x2, x3 = x_[:, 0], x_[:, 1], x_[:, 2]
    O = torch.zeros_like(x1)

    X = torch.stack((
        torch.stack((O, -x3, x2), dim=1),
        torch.stack((x3, O, -x1), dim=1),
        torch.stack((-x2, x1, O), dim=1)), dim=1)
    return X.view(*(x.size()[0:-1]), 3, 3)

def exp(x):
    x_ = x.view(-1, 6)
    w, v = x_[:, 0:3], x_[:, 3:6]
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    W = so3_mat(w)
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

def mat(x):
    # size: [*, 6] -> [*, 4, 4]
    x_ = x.view(-1, 6)
    w1, w2, w3 = x_[:, 0], x_[:, 1], x_[:, 2]
    v1, v2, v3 = x_[:, 3], x_[:, 4], x_[:, 5]
    O = torch.zeros_like(w1)

    X = torch.stack((
        torch.stack((  O, -w3,  w2, v1), dim=1),
        torch.stack(( w3,   O, -w1, v2), dim=1),
        torch.stack((-w2,  w1,   O, v3), dim=1),
        torch.stack((  O,   O,   O,  O), dim=1)), dim=1)
    return X.view(*(x.size()[0:-1]), 4, 4)

def genvec():
    return torch.eye(6)

def genmat():
    return mat(genvec())

class ExpMap(torch.autograd.Function):
    """ Exp: se(3) -> SE(3)
    """
    @staticmethod
    def forward(ctx, x):
        """ Exp: R^6 -> M(4),
            size: [B, 6] -> [B, 4, 4],
              or  [B, 1, 6] -> [B, 1, 4, 4]
        """
        ctx.save_for_backward(x)
        g = exp(x)
        return g

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        g = exp(x)
        gen_k = genmat().to(x)

        # Let z = f(g) = f(exp(x))
        # dz = df/dgij * dgij/dxk * dxk
        #    = df/dgij * (d/dxk)[exp(x)]_ij * dxk
        #    = df/dgij * [gen_k*g]_ij * dxk

        dg = gen_k.matmul(g.view(-1, 1, 4, 4))
        # (k, i, j)
        dg = dg.to(grad_output)

        go = grad_output.contiguous().view(-1, 1, 4, 4)
        dd = go * dg
        grad_input = dd.sum(-1).sum(-1)

        return grad_input

def transform(g, a):
    # g : SE(3),  * x 4 x 4
    # a : R^3,    * x 3[x N]
    g_ = g.view(-1, 4, 4)
    R = g_[:, 0:3, 0:3].contiguous().view(*(g.size()[0:-2]), 3, 3)
    p = g_[:, 0:3, 3].contiguous().view(*(g.size()[0:-2]), 3)
    #print("g")
    #print(g)
    #print("p")
    #print(p)
    if len(g.size()) == len(a.size()):
        b = R.matmul(a) + p.unsqueeze(-1)
    else:
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p
    return b


def mean_shift(template, source, p0_zero_mean, p1_zero_mean):
	template_mean = torch.eye(3).view(1, 3, 3).expand(template.size(0), 3, 3).to(template) 		# [B, 3, 3]
	source_mean = torch.eye(3).view(1, 3, 3).expand(source.size(0), 3, 3).to(source) 			# [B, 3, 3]

	if p0_zero_mean:
		p0_m = template.mean(dim=1) # [B, N, 3] -> [B, 3]
		template_mean = torch.cat([template_mean, p0_m.unsqueeze(-1)], dim=2)
		one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(template_mean.shape[0], 1, 1).to(template_mean)    # (Bx1x4)
		template_mean = torch.cat([template_mean, one_], dim=1)
		template = template - p0_m.unsqueeze(1)
	# else:
		# q0 = template

	if p1_zero_mean:
		#print(numpy.any(numpy.isnan(p1.numpy())))
		p1_m = source.mean(dim=1) # [B, N, 3] -> [B, 3]
		source_mean = torch.cat([source_mean, -p1_m.unsqueeze(-1)], dim=2)
		one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(source_mean.shape[0], 1, 1).to(source_mean)    # (Bx1x4)
		source_mean = torch.cat([source_mean, one_], dim=1)
		source = source - p1_m.unsqueeze(1)
	# else:
		# q1 = source
	return template, source, template_mean, source_mean

def postprocess_data(result, p0, p1, a0, a1, p0_zero_mean, p1_zero_mean):
	#output' = trans(p0_m) * output * trans(-p1_m)
	#        = [I, p0_m;] * [R, t;] * [I, -p1_m;]
	#          [0, 1    ]   [0, 1 ]   [0,  1    ]
	est_g = result['est_T']
	if p0_zero_mean:
		est_g = a0.to(est_g).bmm(est_g)
	if p1_zero_mean:
		est_g = est_g.bmm(a1.to(est_g))
	result['est_T'] = est_g

	est_gs = result['est_T_series'] # [M, B, 4, 4]
	if p0_zero_mean:
		est_gs = a0.unsqueeze(0).contiguous().to(est_gs).matmul(est_gs)
	if p1_zero_mean:
		est_gs = est_gs.matmul(a1.unsqueeze(0).contiguous().to(est_gs))
	result['est_T_series'] = est_gs

	return result

class PointNetLK(nn.Module):
	def __init__(self, feature_model=PointNet(), delta=1.0e-2, learn_delta=False, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True, pooling='max'):
		super().__init__()
		self.feature_model = feature_model
		self.pooling = Pooling(pooling)
		self.inverse = InvMatrix.apply
		self.exp = ExpMap.apply # [B, 6] -> [B, 4, 4]
		self.transform = transform # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

		w1, w2, w3, v1, v2, v3 = delta, delta, delta, delta, delta, delta
		twist = torch.Tensor([w1, w2, w3, v1, v2, v3])
		self.dt = torch.nn.Parameter(twist.view(1, 6), requires_grad=learn_delta)

		# results
		self.last_err = None
		self.g_series = None # for debug purpose
		self.prev_r = None
		self.g = None # estimation result
		self.itr = 0
		self.xtol = xtol
		self.p0_zero_mean = p0_zero_mean
		self.p1_zero_mean = p1_zero_mean

	def forward(self, template, source, maxiter=1000):
		template, source, template_mean, source_mean = mean_shift(template, source, self.p0_zero_mean, self.p1_zero_mean)

		result = self.iclk(template, source, maxiter)
		result = postprocess_data(result, template, source, template_mean, source_mean, self.p0_zero_mean, self.p1_zero_mean)
		return result

	def iclk(self, template, source, maxiter):
		batch_size = template.size(0)

		est_T0 = torch.eye(4).to(template).view(1, 4, 4).expand(template.size(0), 4, 4).contiguous()
		est_T = est_T0
		self.est_T_series = torch.zeros(maxiter+1, *est_T0.size(), dtype=est_T0.dtype)
		self.est_T_series[0] = est_T0.clone()

		training = self.handle_batchNorm(template, source)

		# re-calc. with current modules
		template_features = self.pooling(self.feature_model(template)) # [B, N, 3] -> [B, K]

		# approx. J by finite difference
		dt = self.dt.to(template).expand(batch_size, 6)
		J = self.approx_Jic(template, template_features, dt)
		self.last_err = None
		pinv = self.compute_inverse_jacobian(J, template_features, source)
		
		if pinv == {}:
			result = {'est_R': est_T[:,0:3,0:3],
					  'est_t': est_T[:,0:3,3],
					  'est_T': est_T,
					  'r': None,
					  'transformed_source': self.transform(est_T.unsqueeze(1), source),
					  'itr': 1,
					  'est_T_series': self.est_T_series}
			return result

		itr = 0
		r = None
		for itr in range(maxiter):
			self.prev_r = r
			transformed_source = self.transform(est_T.unsqueeze(1), source) # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
			source_features = self.pooling(self.feature_model(transformed_source)) # [B, N, 3] -> [B, K]
			r = source_features - template_features

			pose = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)

			check = pose.norm(p=2, dim=1, keepdim=True).max()
			if float(check) < self.xtol:
				if itr == 0:
					self.last_err = 0 # no update.
				break

			est_T = self.update(est_T, pose)
			self.est_T_series[itr+1] = est_T.clone()

		rep = len(range(itr, maxiter))
		self.est_T_series[(itr+1):] = est_T.clone().unsqueeze(0).repeat(rep, 1, 1, 1)

		self.feature_model.train(training)
		self.est_T = est_T
		
		#print("est_T.unsqueeze(1)")
		#print(est_T.unsqueeze(1))
		result = {'est_R': est_T[:,0:3,0:3],
				  'est_t': est_T[:,0:3,3],
				  'est_T': est_T,
				  'r': r,
				  'transformed_source': self.transform(est_T.unsqueeze(1), source),
				  'itr': itr+1,
				  'est_T_series': self.est_T_series}

		return result

	def update(self, g, dx):
		# [B, 4, 4] x [B, 6] -> [B, 4, 4]
		dg = self.exp(dx)
		return dg.matmul(g)

	def approx_Jic(self, template, template_features, dt):
		# p0: [B, N, 3], Variable
		# f0: [B, K], corresponding feature vector
		# dt: [B, 6], Variable
		# Jk = (feature_model(p(-delta[k], p0)) - f0) / delta[k]

		batch_size = template.size(0)
		num_points = template.size(1)

		# compute transforms
		transf = torch.zeros(batch_size, 6, 4, 4).to(template)
		for b in range(template.size(0)):
			d = torch.diag(dt[b, :]) # [6, 6]
			D = self.exp(-d) # [6, 4, 4]
			transf[b, :, :, :] = D[:, :, :]
		transf = transf.unsqueeze(2).contiguous()  #   [B, 6, 1, 4, 4]
		p = self.transform(transf, template.unsqueeze(1)) # x [B, 1, N, 3] -> [B, 6, N, 3]

		#f0 = self.feature_model(p0).unsqueeze(-1) # [B, K, 1]
		template_features = template_features.unsqueeze(-1) # [B, K, 1]
		f = self.pooling(self.feature_model(p.view(-1, num_points, 3))).view(batch_size, 6, -1).transpose(1, 2) # [B, K, 6]

		df = template_features - f # [B, K, 6]
		J = df / dt.unsqueeze(1)

		return J

	def compute_inverse_jacobian(self, J, template_features, source):
		# compute pinv(J) to solve J*x = -r
		try:
			Jt = J.transpose(1, 2) # [B, 6, K]
			H = Jt.bmm(J) # [B, 6, 6]
			B = self.inverse(H)
			pinv = B.bmm(Jt) # [B, 6, K]
			return pinv
		except RuntimeError as err:
			# singular...?
			self.last_err = err
			g = torch.eye(4).to(source).view(1, 4, 4).expand(source.size(0), 4, 4).contiguous()
			#print(err)
			# Perhaps we can use MP-inverse, but,...
			# probably, self.dt is way too small...
			source_features = self.pooling(self.feature_model(source)) # [B, N, 3] -> [B, K]
			r = source_features - template_features
			self.feature_model.train(self.feature_model.training)
			return {}

	def handle_batchNorm(self, template, source):
		training = self.feature_model.training
		if training:
			# first, update BatchNorm modules
			template_features, source_features = self.pooling(self.feature_model(template)), self.pooling(self.feature_model(source))
		self.feature_model.eval()	# and fix them.
		return training

def rotation_matrix_to_euler_angles(R):
    """
    Convert a 3x3 rotation matrix to Euler angles.

    Parameters:
        R (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        numpy.ndarray: Euler angles [roll, pitch, yaw] in radians.
    """
    # Extract angles using trigonometric relations
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw])

def coordinate_3d(axes, range_x, range_y, range_z, grid = True):
    axes.set_xlabel("x", fontsize = 14)
    axes.set_ylabel("y", fontsize = 14)
    axes.set_zlabel("z", fontsize = 14)
    axes.set_xlim(range_x[0], range_x[1])
    axes.set_ylim(range_y[0], range_y[1])
    axes.set_zlim(range_z[0], range_z[1])
    if grid == True:
        axes.grid()

def visual_vector_3d(axes, loc, vector, color = "red"):
    axes.quiver(loc[0], loc[1], loc[2],
              vector[0], vector[1], vector[2],
              color = color, lw=3)


###################
#Registration
###################
# Define Registration Algorithm.
def registration_algorithm(device=torch.device('cpu')):

  #reg_algorithm = ICP()

  pretrained_reg = "/home/nishidalab0/MaskNet/pretrained_ptlk/best_model.t7"

  ptnet = PointNet(emb_dims=1024, input_shape="bnc", use_bn=True, global_feat=True)
  pnlk = PointNetLK(feature_model=ptnet, delta=1.0e-2, learn_delta=False, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True, pooling='max')
  if pretrained_reg:
      assert os.path.isfile(pretrained_reg)
      pnlk.load_state_dict(torch.load(pretrained_reg, map_location='cpu'))
      #print("PointNetLK pretrained model loaded successfully!")
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  pnlk = pnlk.to(device)
  reg_algorithm = pnlk

  return reg_algorithm


# Register template and source pairs.
class Registration:
	def __init__(self):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.reg_algorithm = registration_algorithm(device)

	@staticmethod
	def pc2points(data):
		if len(data.shape) == 3:
			return data[:, :, :3]
		elif len(data.shape) == 2:
			return data[:, :3]

	def register(self, template, source):
		# template, source: 		Point Cloud [B, N, 3] (torch tensor)

		# No need to use normals. Only use normals for RPM-Net.
		#if not self.is_rpmnet == 'rpmnet':
		#	template, source = self.pc2points(template), self.pc2points(source)

		result = self.reg_algorithm(template, source)
		return result

def pc2open3d(data):
	if torch.is_tensor(data): data = data.detach().cpu().numpy()
	if len(data.shape) == 2:
		pc = o3d.geometry.PointCloud()
		pc.points = o3d.utility.Vector3dVector(data)
		return pc
	else:
		print("Error in the shape of data given to Open3D!, Shape is ", data.shape)


def display_results_sample(template, source, est_T, masked_template, transformed_source):
  transformed_source = np.matmul(est_T[0:3, 0:3], source.T).T + est_T[0:3, 3]     # ※matmul：行列の積　　第一項：回転、第二項：平行移動
  
  ### x軸の表示 ###
  numpy_ax_x = np.array([[0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0], [0.04, 0, 0], [0.05, 0, 0], [0.06, 0, 0], [0.07, 0, 0], [0.08, 0, 0], [0.09, 0, 0], [0.1, 0, 0]])
  ax_x = o3d.geometry.PointCloud()
  ax_x.points = o3d.utility.Vector3dVector(numpy_ax_x)
  ax_x.paint_uniform_color([1/3, 1/3, 1/3])
		
  ### y軸の表示 ###
  numpy_ax_y = np.array([[0, 0.01, 0], [0, 0.02, 0], [0, 0.03, 0], [0, 0.04, 0], [0, 0.05, 0], [0, 0.06, 0], [0, 0.07, 0], [0, 0.08, 0], [0, 0.09, 0], [0, 0.1, 0]])
  ax_y = o3d.geometry.PointCloud()
  ax_y.points = o3d.utility.Vector3dVector(numpy_ax_y)
  ax_y.paint_uniform_color([1/3, 1/3, 1/3])
		
  ### z軸の表示 ###
  numpy_ax_z = np.array([[0, 0, 0.01], [0, 0, 0.02], [0, 0, 0.03], [0, 0, 0.04], [0, 0, 0.05], [0, 0, 0.06], [0, 0, 0.07], [0, 0, 0.08], [0, 0, 0.09], [0, 0, 0.1]])
  ax_z = o3d.geometry.PointCloud()
  ax_z.points = o3d.utility.Vector3dVector(numpy_ax_z)
  ax_z.paint_uniform_color([1/3, 1/3, 1/3])
  
  ### 原点を表示 ###
  numpy_o = np.array([[0, 0, 0]])
  o = o3d.geometry.PointCloud()
  o.points = o3d.utility.Vector3dVector(numpy_o)
  o.paint_uniform_color([1, 0, 0])
  
  ### 正解を定義 ###
  if theta == 0:
  	## 0度 ##
  	ans_theta_x = np.radians(0)
  	ans_theta_y = np.radians(1)
  	ans_theta_z = np.radians(-184)
  elif theta == 45:
  	## 90度 ##
  	ans_theta_x = np.radians(2)
  	ans_theta_y = np.radians(0.2)
  	ans_theta_z = np.radians(-135)
  elif theta == 90:
  	## 90度 ##
  	ans_theta_x = np.radians(2)
  	ans_theta_y = np.radians(0.2)
  	ans_theta_z = np.radians(-90)
  elif theta == 135:
  	## 90度 ##
  	ans_theta_x = np.radians(2)
  	ans_theta_y = np.radians(0.2)
  	ans_theta_z = np.radians(-45)
  elif theta == "L_90":
  	## 90度 ##
  	ans_theta_x = np.radians(-3)
  	ans_theta_y = np.radians(-1)
  	ans_theta_z = np.radians(-85)
  elif theta == "L_180":
  	## 90度 ##
  	ans_theta_x = np.radians(-12)
  	ans_theta_y = np.radians(-2)
  	ans_theta_z = np.radians(176)
  # x軸方向に回転
  R_x = np.array(
         [[1, 0, 0], 
         [0, np.cos(ans_theta_x), -np.sin(ans_theta_x)], 
         [0, np.sin(ans_theta_x), np.cos(ans_theta_x)]])
  # y軸方向に回転
  R_y = np.array(
         [[np.cos(ans_theta_y), 0, np.sin(ans_theta_y)], 
         [0, 1, 0], 
         [-np.sin(ans_theta_y), 0, np.cos(ans_theta_y)]])
  # z軸方向に回転       
  R_z = np.array(
         [[np.cos(ans_theta_z), -np.sin(ans_theta_z), 0], 
         [np.sin(ans_theta_z), np.cos(ans_theta_z), 0], 
         [0, 0, 1]])
         
  # 回転行列を計算
  ans_R = R_x @ R_y @ R_z
  print("ans_R:\n", ans_R)
  print("est_R:\n", est_T[0:3, 0:3])
  # 平行移動
  ans_t_ = [0, 0.008, -0.011]
  if theta == "L_90":
  
  	ans_t_ = [0.0055, 0.008, -0.011]
  if theta == "L_180":
  
  	ans_t_ = [0.006, 0.008, -0.011]
    
  # 重心移動も含めた変換を行う
  ans_t = np.matmul(ans_R, -np.mean(source, axis=0).T).T + ans_t_
  numpy_ans_source = np.matmul(ans_R, source.T).T + ans_t
  ans_source = o3d.geometry.PointCloud()
  ans_source.points = o3d.utility.Vector3dVector(numpy_ans_source)
  
  ### 回転移動の差分 ###
  euler_angles = rotation_matrix_to_euler_angles(est_T[0:3, 0:3])
  rotation_angle_x = np.degrees(euler_angles[0])
  rotation_angle_y = np.degrees(euler_angles[1])
  rotation_angle_z = np.degrees(euler_angles[2])
  print("\nRotation angle around x-axis:", rotation_angle_x, "degrees")
  print("Rotation angle around y-axis:", rotation_angle_y, "degrees")
  print("Rotation angle around z-axis:", rotation_angle_z, "degrees")
  ###print("\n回転移動の差：")
  diff_R_x = rotation_angle_x - np.degrees(ans_theta_x)
  diff_R_y = rotation_angle_y - np.degrees(ans_theta_y)
  diff_R_z = rotation_angle_z - np.degrees(ans_theta_z)
  ###print("x軸方向　", abs(diff_R_x), " ", "y軸方向　", abs(diff_R_y), " ", "z軸方向　", abs(diff_R_z))
  global diff_R
  diff_R = np.linalg.norm([diff_R_x, diff_R_y, diff_R_z])
  print("\n回転移動の差（L2ノルム）：", diff_R)
  
  ### 平行移動の差分 ###
  global diff_t
  diff_t = np.linalg.norm(est_T[0:3, 3] - ans_t)
  print("平行移動の差（L2ノルム）：", diff_t, "\n")
  print(est_T[0:3, 3])
  print(ans_t )
  
  template = pc2open3d(template)
  source = pc2open3d(source)
  #transformed_source = copy.deepcopy(source)
  #transformed_source.transform(est_T)
  transformed_source = pc2open3d(np.array(transformed_source))
  masked_template = pc2open3d(masked_template)

  template.paint_uniform_color([1, 0, 0])
  source.paint_uniform_color([0, 1, 0])
  transformed_source.paint_uniform_color([0, 1, 0])
  masked_template.paint_uniform_color([0, 0, 1])
  ans_source.paint_uniform_color([1/3, 1/3, 1/3])

  #o3d.visualization.draw_geometries([template])                                    # テンプレ
  o3d.visualization.draw_geometries([masked_template, source, ans_source])          # マスクテンプレ、ソース、正解ソース、原点
  #o3d.visualization.draw_geometries([masked_template, source, source_t, o, ax_x, ax_y, ax_z])
  #o3d.visualization.draw_geometries([masked_template, source, transformed_source])  # マスクテンプレ、ソース、変換後ソース
  o3d.visualization.draw_geometries([template, source, transformed_source, o])        # テンプレ、ソース、変換後ソース
  #o3d.visualization.draw_geometries([template, masked_template, source])           # テンプレ、マスクテンプレ、ソース
  #o3d.visualization.draw_geometries([template, source])                            # テンプレ、ソース
  #o3d.visualization.draw_geometries([masked_template, source])                     # マスクテンプレ、ソース
  ###masked_template.paint_uniform_color([0, 1, 0])
  ###o3d.visualization.draw_geometries([masked_template, source])                     # マスクテンプレ（green）
