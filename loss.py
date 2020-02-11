from torch import nn
from torch.distributions.categorical import Categorical

import torch
import numpy as np


class LigthSpeechLoss(nn.Module):
    """ LigthSpeech Loss """

    def __init__(self):
        super(LigthSpeechLoss, self).__init__()

    def forward(self, mel, predicted_length, mel_target, length_target, P, history):
        mel_loss = nn.MSELoss()(mel, mel_target)
        len_loss = nn.L1Loss()(predicted_length.float(), length_target.float())

        # print(predicted_length)
        # print(length_target)

        # similarity_loss = nn.MSELoss()(cemb_out, cemb)
        # similarity_loss = nn.L1Loss()(cemb_out, cemb)

        # duration_loss = nn.L1Loss()(padd_predicted, D.float())

        # ------------ Policy Gradient ------------ #

        rewards = list()
        for batch_ind in range(mel.size(0)):
            len_cut = length_target[batch_ind]
            mel_target_cut = mel_target[batch_ind][:len_cut]
            mel_pred_cut = mel[batch_ind][:len_cut]
            mat = 1.0 / (torch.pow(mel_pred_cut-mel_target_cut, 2) + 1.0)
            rewards.append(torch.sum(torch.sum(mat, -1), -1).item())
        # rewards = torch.Tensor(rewards).to("cuda").reshape((-1, 1, 1))
        rewards = torch.Tensor(rewards).to("cuda").reshape((-1, 1))

        rewards = (rewards - rewards.mean()) / \
            (rewards.std() + np.finfo(np.float32).eps)

        # pg_loss = torch.sum(torch.sum(
        #     torch.sum(torch.mul(rewards, P).mul(-1), -1), -1), -1) / rewards.size(0)
        pg_loss = torch.sum(torch.sum(
            torch.sum(torch.mul(rewards, history).mul(-1), -1), -1), -1) / rewards.size(0)

        # ------------ Policy Gradient ------------ #

        # return mel_loss, similarity_loss, duration_loss
        return mel_loss, len_loss, pg_loss
