# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker

from torch.autograd import Variable
import torch


class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def _rtaa_attack(self, x_init, x, scale_z, eps=10, alpha=1, iteration=10, x_val_min=0, x_val_max=255):
        """ fsgm attack """

        x = Variable(x.data)
        x_adv = Variable(x_init.data, requires_grad=True)

        alpha = eps * 1.0 / iteration

        for i in range(iteration):

            all_result = self.model.track(x_adv)
            score = all_result['cls']
            delta = all_result['loc']

            gt = np.array([self.center_pos[0] - self.size[0] / 2, self.center_pos[1] - self.size[1] / 2, self.size[0],
                           self.size[1]])

            score_temp = score.permute(1, 2, 3, 0).contiguous().view(2, -1)
            score = torch.transpose(score_temp, 0, 1)
            delta1 = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
            delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()

            # calculate proposals
            gt_cen = np.array([self.center_pos[0], self.center_pos[1], self.size[0], self.size[1]])
            gt_cen = np.tile(gt_cen, (self.anchors.shape[0], 1))
            gt_cen[:, 0] = ((gt_cen[:, 0] - self.center_pos[0]) * scale_z - self.anchors[:, 0]) / self.anchors[:, 2]
            gt_cen[:, 1] = ((gt_cen[:, 1] - self.center_pos[1]) * scale_z - self.anchors[:, 1]) / self.anchors[:, 3]
            gt_cen[:, 2] = np.log(gt_cen[:, 2] * scale_z) / self.anchors[:, 2]
            gt_cen[:, 3] = np.log(gt_cen[:, 3] * scale_z) / self.anchors[:, 3]

            # create pseudo proposals randomly
            gt_cen_pseudo = rect_2_cxy_wh(gt)
            gt_cen_pseudo = np.tile(gt_cen_pseudo, (self.anchors.shape[0], 1))

            rate_xy1 = np.random.uniform(0.3, 0.5)
            rate_xy2 = np.random.uniform(0.3, 0.5)
            rate_wd = np.random.uniform(0.7, 0.9)

            gt_cen_pseudo[:, 0] = ((gt_cen_pseudo[:, 0] - self.center_pos[0] - rate_xy1 * gt_cen_pseudo[:,
                                                                                          2]) * scale_z - self.anchors[
                                                                                                          :,
                                                                                                          0]) / self.anchors[
                                                                                                                :, 2]
            gt_cen_pseudo[:, 1] = ((gt_cen_pseudo[:, 1] - self.center_pos[1] - rate_xy2 * gt_cen_pseudo[:,
                                                                                          3]) * scale_z - self.anchors[
                                                                                                          :,
                                                                                                          1]) / self.anchors[
                                                                                                                :, 3]
            gt_cen_pseudo[:, 2] = np.log(gt_cen_pseudo[:, 2] * rate_wd * scale_z) / self.anchors[:, 2]
            gt_cen_pseudo[:, 3] = np.log(gt_cen_pseudo[:, 3] * rate_wd * scale_z) / self.anchors[:, 3]

            delta[0, :] = (delta[0, :] * self.anchors[:, 2] + self.anchors[:, 0]) / scale_z + self.center_pos[0]
            delta[1, :] = (delta[1, :] * self.anchors[:, 3] + self.anchors[:, 1]) / scale_z + self.center_pos[1]
            delta[2, :] = (np.exp(delta[2, :]) * self.anchors[:, 2]) / scale_z
            delta[3, :] = (np.exp(delta[3, :]) * self.anchors[:, 3]) / scale_z
            location = np.array([delta[0] - delta[2] / 2, delta[1] - delta[3] / 2, delta[2], delta[3]])

            label = overlap_ratio(location, gt)

            # set thresold to define positive and negative samples, following the training step
            iou_hi = 0.6
            iou_low = 0.3

            # make labels
            y_pos = np.where(label > iou_hi, 1, 0)
            y_pos = torch.from_numpy(y_pos).cuda().long()
            y_neg = np.where(label < iou_low, 0, 1)
            y_neg = torch.from_numpy(y_neg).cuda().long()
            pos_index = np.where(y_pos.cpu() == 1)
            neg_index = np.where(y_neg.cpu() == 0)
            index = np.concatenate((pos_index, neg_index), axis=1)

            # make pseudo lables
            y_pos_pseudo = np.where(label > iou_hi, 0, 1)
            y_pos_pseudo = torch.from_numpy(y_pos_pseudo).cuda().long()
            y_neg_pseudo = np.where(label < iou_low, 1, 0)
            y_neg_pseudo = torch.from_numpy(y_neg_pseudo).cuda().long()

            y_truth = y_pos
            y_pseudo = y_pos_pseudo

            # calculate classification loss
            loss_truth_cls = -F.cross_entropy(score[index], y_truth[index])
            loss_pseudo_cls = -F.cross_entropy(score[index], y_pseudo[index])
            loss_cls = (loss_truth_cls - loss_pseudo_cls) * (1)

            # calculate regression loss
            loss_truth_reg = -rpn_smoothL1(delta1, gt_cen, y_pos)
            loss_pseudo_reg = -rpn_smoothL1(delta1, gt_cen_pseudo, y_pos)
            loss_reg = (loss_truth_reg - loss_pseudo_reg) * (5)

            # final adversarial loss
            loss = loss_cls  + loss_reg

            # calculate the derivative
            self.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            loss.backward(retain_graph=True)

            adv_grad = where((x_adv.grad > 3e-6) | (x_adv.grad < -3e-6), x_adv.grad, 0)
            adv_grad = torch.sign(adv_grad)
            x_adv = x_adv - alpha * adv_grad

            x_adv = where(x_adv > x + eps, x + eps, x_adv)
            x_adv = where(x_adv < x - eps, x - eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)


        return x_adv

    def _rtaa_defense(self, x_mask, x, x_crop, scale_z, eps=5, weight=10, alpha=1, iteration=10, x_val_min=0, x_val_max=255):
        """ fsgm defense """

        x = Variable(x.data)
        x_adv = Variable(x_mask.data, requires_grad=True)

        alpha = eps * 1.0 / iteration

        for i in range(iteration):

            all_result = self.model.track(x_adv)
            score = all_result['cls']
            delta = all_result['loc']

            gt = np.array([self.center_pos[0] - self.size[0] / 2, self.center_pos[1] - self.size[1] / 2, self.size[0],
                           self.size[1]])

            score_temp = score.permute(1, 2, 3, 0).contiguous().view(2, -1)
            score = torch.transpose(score_temp, 0, 1)
            delta1 = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
            delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()

            # calculate proposals
            gt_cen = np.array([self.center_pos[0], self.center_pos[1], self.size[0], self.size[1]])
            gt_cen = np.tile(gt_cen, (self.anchors.shape[0], 1))
            gt_cen[:, 0] = ((gt_cen[:, 0] - self.center_pos[0]) * scale_z - self.anchors[:, 0]) / self.anchors[:, 2]
            gt_cen[:, 1] = ((gt_cen[:, 1] - self.center_pos[1]) * scale_z - self.anchors[:, 1]) / self.anchors[:, 3]
            gt_cen[:, 2] = np.log(gt_cen[:, 2] * scale_z) / self.anchors[:, 2]
            gt_cen[:, 3] = np.log(gt_cen[:, 3] * scale_z) / self.anchors[:, 3]

            # create pseudo proposals randomly
            gt_cen_pseudo = rect_2_cxy_wh(gt)
            gt_cen_pseudo = np.tile(gt_cen_pseudo, (self.anchors.shape[0], 1))

            rate_xy1 = np.random.uniform(0.3, 0.5)
            rate_xy2 = np.random.uniform(0.3, 0.5)
            rate_wd = np.random.uniform(0.7, 0.9)

            gt_cen_pseudo[:, 0] = ((gt_cen_pseudo[:, 0] - self.center_pos[0] - rate_xy1 * gt_cen_pseudo[:,
                                                                                          2]) * scale_z - self.anchors[
                                                                                                          :,
                                                                                                          0]) / self.anchors[
                                                                                                                :, 2]
            gt_cen_pseudo[:, 1] = ((gt_cen_pseudo[:, 1] - self.center_pos[1] - rate_xy2 * gt_cen_pseudo[:,
                                                                                          3]) * scale_z - self.anchors[
                                                                                                          :,
                                                                                                          1]) / self.anchors[
                                                                                                                :, 3]
            gt_cen_pseudo[:, 2] = np.log(gt_cen_pseudo[:, 2] * rate_wd * scale_z) / self.anchors[:, 2]
            gt_cen_pseudo[:, 3] = np.log(gt_cen_pseudo[:, 3] * rate_wd * scale_z) / self.anchors[:, 3]

            delta[0, :] = (delta[0, :] * self.anchors[:, 2] + self.anchors[:, 0]) / scale_z + self.center_pos[0]
            delta[1, :] = (delta[1, :] * self.anchors[:, 3] + self.anchors[:, 1]) / scale_z + self.center_pos[1]
            delta[2, :] = (np.exp(delta[2, :]) * self.anchors[:, 2]) / scale_z
            delta[3, :] = (np.exp(delta[3, :]) * self.anchors[:, 3]) / scale_z
            location = np.array([delta[0] - delta[2] / 2, delta[1] - delta[3] / 2, delta[2], delta[3]])

            label = overlap_ratio(location, gt)

            # set thresold to define positive and negative samples, following the training step
            iou_hi = 0.6
            iou_low = 0.3

            # make labels
            y_pos = np.where(label > iou_hi, 1, 0)
            y_pos = torch.from_numpy(y_pos).cuda().long()
            y_neg = np.where(label < iou_low, 0, 1)
            y_neg = torch.from_numpy(y_neg).cuda().long()
            pos_index = np.where(y_pos.cpu() == 1)
            neg_index = np.where(y_neg.cpu() == 0)
            index = np.concatenate((pos_index, neg_index), axis=1)

            # make pseudo lables
            y_pos_pseudo = np.where(label > iou_hi, 0, 1)
            y_pos_pseudo = torch.from_numpy(y_pos_pseudo).cuda().long()
            y_neg_pseudo = np.where(label < iou_low, 1, 0)
            y_neg_pseudo = torch.from_numpy(y_neg_pseudo).cuda().long()

            y_truth = y_pos
            y_pseudo = y_pos_pseudo

            # calculate classification loss
            loss_truth_cls = -F.cross_entropy(score[index], y_truth[index])
            loss_pseudo_cls = -F.cross_entropy(score[index], y_pseudo[index])
            loss_cls = (loss_truth_cls - loss_pseudo_cls) * (1)

            # calculate regression loss
            loss_truth_reg = -rpn_smoothL1(delta1, gt_cen, y_pos)
            loss_pseudo_reg = -rpn_smoothL1(delta1, gt_cen_pseudo, y_pos)
            loss_reg = (loss_truth_reg - loss_pseudo_reg) * (5)

            # final adversarial loss
            loss = loss_cls + loss_reg

            # calculate the derivative
            self.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            loss.backward(retain_graph=True)


            # defence
            mask = x_adv.grad * weight
            mask = where(mask > alpha, (alpha + 0.01), mask)
            mask = where(mask < -alpha, (-alpha - 0.01), mask)
            x_adv = x_adv + mask

            x_adv = where(x_adv > x + eps, x + eps, x_adv)
            x_adv = where(x_adv < x - eps, x - eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)


        return x_adv

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])


        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        # print(best_idx)
        return {
            'bbox': bbox,
            'best_score': best_score,
            'crop': x_crop
        }

    def track_att(self, img, att_per):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        # adversarial attack
        x_crop_init = x_crop + att_per * 1
        x_crop_init = torch.clamp(x_crop_init, 0, 255)
        x_adv1 = self._rtaa_attack(x_crop_init, x_crop, scale_z)
        att_per = x_adv1 - x_crop

        outputs = self.model.track(x_adv1)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])


        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                   'bbox': bbox,
                   'best_score': best_score,
               }, att_per

    def track_attdef(self, img, att_per, def_per):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        # adversarial attack
        x_crop_init = x_crop + att_per * 1
        x_crop_init = torch.clamp(x_crop_init, 0, 255)
        x_adv1 = self._rtaa_attack(x_crop_init, x_crop, scale_z, iteration=10)
        att_per = x_adv1 - x_crop


        # adversarial defense
        x_adv2_mask = x_adv1 + def_per * 0.01
        x_adv2_mask = torch.clamp(x_adv2_mask, 0, 255)
        x_adv2 = self._rtaa_defense(x_adv2_mask, x_adv1, x_crop, scale_z, iteration=10, weight=1000)
        def_per = x_adv2 - x_adv1

        outputs = self.model.track(x_adv2)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])


        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {
                   'bbox': bbox,
                   'best_score': best_score,
               }, att_per, def_per


def rpn_smoothL1(input, target, label):
    r"""
    :param input: torch.Size([1, 1125, 4])
    :param target: torch.Size([1, 1125, 4])
            label: (torch.Size([1, 1125]) pos neg or ignore
    :return:
    """
    input = torch.transpose(input, 0, 1)
    pos_index = np.where(label.cpu() == 1)
    target = torch.from_numpy(target).cuda().float()
    loss = F.smooth_l1_loss(input[pos_index], target[pos_index], reduction='sum')

    return loss

def rect_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2, rect[2], rect[3]])


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''
    rect1 = np.transpose(rect1)

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]


    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)