import torch
import torch.nn as nn
import torch.nn.functional as F

class Pooling(torch.nn.Module):
	def __init__(self, pool_type='max'):
		self.pool_type = pool_type
		super(Pooling, self).__init__()

	def forward(self, input):
		if self.pool_type == 'max':
			return torch.max(input, 2)[0].contiguous()
		elif self.pool_type == 'avg' or self.pool_type == 'average':
			return torch.mean(input, 2).contiguous()
			
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


if __name__ == '__main__':
	# Test the code.
	x = torch.rand((10,1024,3))

	pn = PointNet(use_bn=True)
	y = pn(x)
	print("Network Architecture: ")
	print(pn)
	print("Input Shape of PointNet: ", x.shape, "\nOutput Shape of PointNet: ", y.shape)

	class PointNet_modified(PointNet):
		def __init__(self):
			super().__init__()

		def create_structure(self):
			self.conv1 = torch.nn.Conv1d(3, 64, 1)
			self.conv2 = torch.nn.Conv1d(64, 128, 1)
			self.conv3 = torch.nn.Conv1d(128, self.emb_dims, 1)
			self.relu = torch.nn.ReLU()

			layers = [self.conv1, self.relu,
					  self.conv2, self.relu,
					  self.conv3, self.relu]
			return layers

	pn = PointNet_modified()
	y = pn(x)
	print("\n\n\nModified Network Architecture: ")
	print(pn)
	print("Input Shape of PointNet: ", x.shape, "\nOutput Shape of PointNet: ", y.shape)

class PointNetMask(nn.Module):
	def __init__(self, template_feature_size=1024, source_feature_size=1024, feature_model=PointNet()):
		super().__init__()
		self.feature_model = feature_model
		self.pooling = Pooling()

		input_size = template_feature_size + source_feature_size
		self.h3 = nn.Sequential(nn.Conv1d(input_size, 1024, 1), nn.ReLU(),
								nn.Conv1d(1024, 512, 1), nn.ReLU(),
								nn.Conv1d(512, 256, 1), nn.ReLU(),
								nn.Conv1d(256, 128, 1), nn.ReLU(),
								nn.Conv1d(128,   1, 1), nn.Sigmoid())

	def find_mask(self, x, t_out_h1):
		batch_size, _ , num_points = t_out_h1.size()
		x = x.unsqueeze(2)
		x = x.repeat(1,1,num_points)
		x = torch.cat([t_out_h1, x], dim=1)
		x = self.h3(x)
		return x.view(batch_size, -1)

	def forward(self, template, source):
		source_features = self.feature_model(source)				# [B x C x N]
		template_features = self.feature_model(template)			# [B x C x N]

		source_features = self.pooling(source_features)
		mask = self.find_mask(source_features, template_features)
		return mask


class MaskNet(nn.Module):
	def __init__(self, feature_model=PointNet(use_bn=True), is_training=True):
		super().__init__()
		self.maskNet = PointNetMask(feature_model=feature_model)
		self.is_training = is_training

	@staticmethod
	def index_points(points, idx):
		"""
		Input:
			points: input points data, [B, N, C]
			idx: sample index data, [B, S]
		Return:
			new_points:, indexed points data, [B, S, C]
		"""
		device = points.device
		B = points.shape[0]
		view_shape = list(idx.shape)
		view_shape[1:] = [1] * (len(view_shape) - 1)
		repeat_shape = list(idx.shape)
		repeat_shape[0] = 1
		batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
		new_points = points[batch_indices, idx, :]
		return new_points

	# This function is only useful for testing with a single pair of point clouds.
	@staticmethod
	def find_index(mask_val):
		mask_idx = torch.nonzero((mask_val[0]>0.5)*1.0)
		return mask_idx.view(1, -1)

	def forward(self, template, source, point_selection='threshold'):
		mask = self.maskNet(template, source)

		if point_selection == 'topk' or self.is_training:
			_, self.mask_idx = torch.topk(mask, source.shape[1], dim=1, sorted=False)
		elif point_selection == 'threshold':
			self.mask_idx = self.find_index(mask)

		template = self.index_points(template, self.mask_idx)
		return template, mask


if __name__ == '__main__':
	template, source = torch.rand(10,1024,3), torch.rand(10,1024,3)
	net = MaskNet()
	result = net(template, source)
	#import ipdb; ipdb.set_trace()

