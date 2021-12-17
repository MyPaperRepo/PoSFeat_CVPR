import torch
import torch.nn as nn
import torch.nn.functional as F
# from . import preprocess_utils as putils
from .preprocess_utils import *

__all__=['Preprocess_Grid_Line2window', 'Preprocess_full_epipolar']

class Preprocess_Grid_Line2window(nn.Module):
    '''
    the preprocess class for grid-with-line pipeline
    '''
    def __init__(self, configs, device=None, vis=False):
        super(Preprocess_Grid_Line2window, self).__init__()
        self.__lossname__ = 'Preprocess_Grid_Line2window'
        self.config = configs
        self.kps_generator = generate_kpts_regular_grid_random
        # self.kps_generator = getattr(putils, self.config['kps_generator'])
        self.t_base = self.config['temperature_base']
        self.t_max = self.config['temperature_max']
        if device is not None:
            self.device = device

    def name(self):
        return self.__lossname__

    def forward(self, inputs, outputs):
        preds1 = outputs['preds1']
        preds2 = outputs['preds2']

        xc1, xf1 = preds1['global_map'], preds1['local_map']
        xc2, xf2 = preds2['global_map'], preds2['local_map']
        h1i, w1i = inputs['im1'].size()[2:]
        h2i, w2i = inputs['im2'].size()[2:]
        b, _, hf, wf = xf1.shape
        temperature = min(self.t_base + outputs['epoch'], self.t_max)

        # first, we search locate the correspondence with grid points
        # with keep_spatial==True, coord (score) is with bxhxwx2 (bxhxwx1)
        # with keep_spatial==False, coord (score) is with bx(h*w)x2 (bx(h*w)x1)
        # the keep_spatial is defined in self.config['kps_generator_config']
        coord1_n, coord2_n, score1, score2 = self.kps_generator(inputs, outputs, **self.config['kps_generator_config'])
        _, hkps, wkps, _ = coord1_n.shape
        coord1 = denormalize_coords(coord1_n.reshape(b,-1,2), h1i, w1i)
        coord2 = denormalize_coords(coord2_n.reshape(b,-1,2), h2i, w2i)

        feat1_fine = sample_feat_by_coord(xf1, coord1_n.reshape(b,-1,2), self.config['loss_distance']=='cos')
        feat2_fine = sample_feat_by_coord(xf2, coord2_n.reshape(b,-1,2), self.config['loss_distance']=='cos')

        feat1_c_corloc_n_, feat1_c_corloc_n_org, valid1, epi_std1 = epipolar_line_search(coord1, inputs['F1'], feat1_fine, 
            temperature*F.normalize(xf2,p=2.0,dim=1), h2i, w2i, window_size=self.config['window_size'], **self.config['line_search_config'])
        feat2_c_corloc_n_, feat2_c_corloc_n_org, valid2, epi_std2 = epipolar_line_search(coord2, inputs['F2'], feat2_fine, 
            temperature*F.normalize(xf1,p=2.0,dim=1), h1i, w1i, window_size=self.config['window_size'], **self.config['line_search_config'])
        feat1c_corloc_org = denormalize_coords(feat1_c_corloc_n_org, h2i, w2i)
        feat2c_corloc_org = denormalize_coords(feat2_c_corloc_n_org, h1i, w1i)

        feat1w_corloc_n, window_coords_n_1in2, feat1w_std, _ = get_expected_correspondence_within_window(
            feat1_fine, temperature*F.normalize(xf2,p=2.0,dim=1), feat1_c_corloc_n_, self.config['window_size'], with_std=True)
        feat2w_corloc_n, window_coords_n_2in1, feat2w_std, _ = get_expected_correspondence_within_window(
            feat2_fine, temperature*F.normalize(xf1,p=2.0,dim=1), feat2_c_corloc_n_, self.config['window_size'], with_std=True)

        feat1w_corloc = denormalize_coords(feat1w_corloc_n, h2i, w2i)
        feat2w_corloc = denormalize_coords(feat2w_corloc_n, h1i, w1i)

        #  we disabled this, therefore, the values are all one.
        mc_num1 = torch.ones_like(feat1w_std)
        mc_num2 = torch.ones_like(feat2w_std)
        return {
                'coord1':coord1, 'coord2':coord2,
                'feat1w_corloc':feat1w_corloc,
                'feat2w_corloc':feat2w_corloc,
                'feat1c_corloc_org':feat1c_corloc_org,
                'feat2c_corloc_org':feat2_c_corloc_n_org,
                'feat1w_std':feat1w_std, 'feat2w_std':feat2w_std,
                'mc_num1':mc_num1, 'mc_num2':mc_num2,
                'temperature':temperature,
                'valid_epi1':valid1, 'valid_epi2':valid2
                }

class Preprocess_full_epipolar(nn.Module):
    '''
    the preprocess class for joint keypoint detection pipeline with full epipolar distance
    '''
    def __init__(self, configs, device=None, vis=False):
        super(Preprocess_full_epipolar, self).__init__()
        self.__lossname__ = 'Preprocess_full_epipolar'
        self.config = configs
        self.kps_generator = generate_kpts_regular_grid_random
        self.matcher = OT_sinkhorn_log
        self.vis=vis
        self.t_base = self.config['temperature_base']
        self.t_max = self.config['temperature_max']
        if device is not None:
            self.device = device

    def name(self):
        return self.__lossname__

    def forward(self, inputs, outputs):
        coord1_n, coord2_n, score1, score2 = self.kps_generator(inputs, outputs, **self.config['kps_generator_config'])
        # coord1, coord2, score1, score2 = self.kps_generator(inputs, outputs, **self.config['kps_generator_config'])

        preds1 = outputs['preds1']
        preds2 = outputs['preds2']

        xc1, xf1 = preds1['global_map'], preds1['local_map']
        xc2, xf2 = preds2['global_map'], preds2['local_map']
        h1i, w1i = inputs['im1'].size()[2:]
        h2i, w2i = inputs['im2'].size()[2:]
        coord1 = denormalize_coords(coord1_n, h1i, w1i)
        coord2 = denormalize_coords(coord2_n, h2i, w2i)
        # coord1_n = normalize_coords(coord1, h1i, w1i)
        # coord2_n = normalize_coords(coord2, h2i, w2i)

        # process the features in coarse level
        # feat1_coarse = sample_feat_by_coord(xc1, coord1_n, self.config['loss_distance']=='cos')  # Bxnxd
        # feat1_c_corloc_n, feat1_c_std, feat1_c_peak, feat1_c_prob = get_expected_correspondence_locs(
        #     feat1_coarse, xc2.detach(), with_std=True)

        # feat2_coarse = sample_feat_by_coord(xc2, coord2_n, self.config['loss_distance']=='cos')  # Bxnxd
        # feat2_c_corloc_n, feat2_c_std, feat2_c_peak, feat2_c_prob = get_expected_correspondence_locs(
        #     feat2_coarse, xc1.detach(), with_std=True)

        # process the features in fine level
        feat1_fine = sample_feat_by_coord(xf1, coord1_n, self.config['loss_distance']=='cos') #bxn1xc
        feat2_fine = sample_feat_by_coord(xf2, coord2_n, self.config['loss_distance']=='cos') #bxn2xc
        sim_mat = feat1_fine@feat2_fine.transpose(1,2) # bxn1xn2
        cost_mat = 1 - sim_mat
        temperature = min(self.t_base + outputs['epoch'], self.t_max)
        cor_mat, cor_mat_unmatch = self.matcher(cost_mat, temperature=temperature, **self.config['matcher_config'])

        if cor_mat_unmatch is not None:
            unmatch1 = cor_mat_unmatch[:,:-1,-1].unsqueeze(2) # bxn1x1
            unmatch2 = cor_mat_unmatch[:,-1,:-1].unsqueeze(1) # bx1xn2
            valid1 = cor_mat.max(2,True)[0]>unmatch1
            valid2 = cor_mat.max(1,True)[0]>unmatch2
        else:
            # use double check to define valid correspondence
            b, m, n = cor_mat.shape
            nn12 = torch.max(cor_mat, dim=2)[1]
            nn21 = torch.max(cor_mat, dim=1)[1]
            ids1 = torch.arange(0, m, device=cor_mat.device)[None, :].repeat(b, 1)
            ids2 = torch.arange(0, n, device=cor_mat.device)[None, :].repeat(b, 1)
            valid1 = ids1 == nn21.gather(dim=1, index=nn12)
            valid2 = ids2 == nn12.gather(dim=1, index=nn21)

            unmatch1 = torch.zeros(b,m,1).to(cor_mat)
            unmatch2 = torch.zeros(b,1,n).to(cor_mat)

        im_size1 = inputs['im1'].size()[2:]
        im_size2 = inputs['im2'].size()[2:]
        coord1_h = homogenize(coord1).transpose(1, 2) #bx3xm
        coord2_h = homogenize(coord2).transpose(1, 2) #bx3xn
        fmatrix = inputs['F1']
        fmatrix2 = inputs['F2']
        epipolar_line = fmatrix.bmm(coord1_h)
        epipolar_line_ = epipolar_line / torch.clamp(
            torch.norm(epipolar_line[:, :2, :], p=2, dim=1, keepdim=True), min=1e-8)
        valid_epi = valid_points(epipolar_line, im_size1, self.config['linelen_thr'])
        epipolar_dist = torch.abs(epipolar_line_.transpose(1, 2)@coord2_h) #bxmxn

        epipolar_line2 = fmatrix2.bmm(coord2_h)
        epipolar_line2_ = epipolar_line2 / torch.clamp(
            torch.norm(epipolar_line2[:, :2, :], p=2, dim=1, keepdim=True), min=1e-8)
        valid_epi2 = valid_points(epipolar_line2, im_size2, self.config['linelen_thr'])
        epipolar_dist2 = torch.abs(epipolar_line2_.transpose(1, 2)@coord1_h) #bxnxm
        epipolar_dist2 = epipolar_dist2.transpose(1,2) #bxmxn

        return {'coord1':coord1, 'coord2':coord2, # bxmx2, bxnx2
                'score1':score1, 'score2':score2, # bxmx1, bxnx1
                'valid_nn1':valid1, 'valid_nn2':valid2, # bxm, bxn
                'valid_epi1': valid_epi, 'valid_epi2': valid_epi2, #bxm
                'unmatch1': unmatch1, 'unmatch2': unmatch2, # bxmx1, bx1xn
                'sim_mat':sim_mat, 'cor_mat':cor_mat, # bxmxn, bxmxn
                'dist_mat1':epipolar_dist, 'dist_mat2':epipolar_dist2, # bxmxn # bxmxn
                'temperature':temperature
                }