import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np
from collections import OrderedDict

import hparams as hp
import utils


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='relu'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = x .contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor()

    def LR(self, x, duration_predictor_output, alpha=1.0, mel_max_length=None):
        output = list()
        mel_length = list()

        for batch, expand_target in zip(x, duration_predictor_output):
            one_batch = self.expand(batch, expand_target, alpha)
            output.append(one_batch)
            mel_length.append(one_batch.size(0))

        if mel_max_length:
            output = utils.pad(output, mel_max_length)
        else:
            output = utils.pad(output)

        return output, mel_length

    def expand(self, batch, predicted, alpha):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size*alpha), -1))
        out = torch.cat(out, 0)

        return out

    def rounding(self, num):
        if num - int(num) >= 0.5:
            return int(num) + 1
        else:
            return int(num)

    def transfer(self, duration_predictor_P, length_c=None):
        if self.training:
            total_len = duration_predictor_P.size(1)
            # mat = list()
            output = list()
            history = list()
            for i in range(duration_predictor_P.size(0)):
                batch = list()

                # print(duration_predictor_P[i])
                m = Categorical(duration_predictor_P[i])
                history.append(m.log_prob(m.sample()))

                for j in range(total_len):
                    c = Categorical(duration_predictor_P[i][j])
                    c_sample = c.sample() + 1

                    if j < length_c[i]:
                        batch.append(c_sample)
                    else:
                        batch.append(0)

                output.append(batch)

                # batch = [[i for i in range(duration_predictor_P.size(-1))] for _ in range(length_c[i])] + [
                #     [0 for _ in range(duration_predictor_P.size(-1))] for _ in range(total_len-length_c[i])]
                # mat.append(torch.Tensor(batch))
            # mat = torch.stack(mat).to("cuda")
            # return torch.sum(torch.mul(mat, duration_predictor_P), -1).int()

            return torch.Tensor(output).int().to("cuda"), torch.stack(history).float()
        else:
            return torch.max(duration_predictor_P, -1)[1]

    def forward(self, x, length_c=None, alpha=1.0, mel_max_length=None):
        if self.training:
            # print(x)
            duration_predictor_P = self.duration_predictor(x)
            duration_predictor_output, history = self.transfer(duration_predictor_P,
                                                               length_c=length_c)
            output, mel_length = self.LR(x, duration_predictor_output, alpha)
            return output, duration_predictor_P, mel_length, history
        else:
            duration_predictor_P = self.duration_predictor(x)
            duration_predictor_output = self.transfer(duration_predictor_P,
                                                      length_c=length_c)
            output, mel_length = self.LR(x, duration_predictor_output, alpha)
            return output


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self):
        super(DurationPredictor, self).__init__()

        self.linear_layer_1 = Linear(
            hp.pre_gru_out_dim, hp.pre_gru_out_dim, w_init='relu')
        self.relu_1 = nn.ReLU()
        self.linear_layer_2 = Linear(hp.pre_gru_out_dim,
                                     hp.P_dim,
                                     w_init='linear')
        # self.relu_2 = nn.ReLU()

        # predict P
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.linear_layer_1(x)
        out = self.relu_1(out)
        out = self.linear_layer_2(out)
        # out = self.relu_2(out)

        # out = out.squeeze()
        # if not self.training:
        #     out = out.unsqueeze(0)
        out = self.softmax(out)

        return out


class PostNet(nn.Module):
    """ Post Net (Not Add Dropout) """

    def __init__(self):
        super(PostNet, self).__init__()
        self.gru_1 = nn.GRU(hp.n_mel_channels,
                            hp.n_mel_channels,
                            num_layers=1,
                            batch_first=True)

        # self.linear = Linear(hp.n_mel_channels,
        #                      hp.n_mel_channels)

        self.gru_2 = nn.GRU(hp.n_mel_channels,
                            hp.n_mel_channels,
                            num_layers=1,
                            batch_first=True)

        self.dropout = nn.Dropout(0.1)

    def mask(self, mel_1, mel_2, length_mel, max_mel_len):
        x_mask = ~utils.get_mask_from_lengths(length_mel, max_mel_len)
        x_mask = x_mask.expand(hp.n_mel_channels,
                               x_mask.size(0), x_mask.size(1))
        x_mask = x_mask.permute(1, 2, 0)
        mel_1.data.masked_fill_(x_mask, 0.0)
        mel_2.data.masked_fill_(x_mask, 0.0)

        return mel_1, mel_2

    def forward(self, mels, length_mel, max_mel_len):
        self.gru_1.flatten_parameters()
        x, _ = self.gru_1(mels)

        mel_postnet_1 = mels + x
        self.gru_2.flatten_parameters()
        y, _ = self.gru_2(mel_postnet_1)

        mel_postnet_2 = mel_postnet_1 + x + y
        mel_postnet_1, mel_postnet_2 = self.mask(
            mel_postnet_1, mel_postnet_2, length_mel, max_mel_len)

        return mel_postnet_1, mel_postnet_2

    def inference(self, mels):
        x, _ = self.gru_1(mels)
        mel_postnet_1 = mels + x
        y, _ = self.gru_2(mel_postnet_1)
        mel_postnet_2 = mel_postnet_1 + x + y

        return mel_postnet_1, mel_postnet_2


if __name__ == "__main__":
    # Test
    test_dp = DurationPredictor()
    print(utils.get_param_num(test_dp))
