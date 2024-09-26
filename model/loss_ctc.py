import torch
from torch import nn, Tensor


class CTCLoss(nn.Module):
    def __init__(self, label_size=None):
        super(CTCLoss, self).__init__()
        self.label_size = label_size

    def forward(self, scores: Tensor, labels: Tensor) -> Tensor:
        if len(scores.shape) == 1:
            label = labels.to(scores.device)
            score = scores

            # 取出负例-正例的差值
            matrix_output = score[:, None] - score[None, :]  # 这里是算出所有位置 两两之间余弦的差值
            # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
            matrix_label = label[:, None] < label[None, :]  # 取出负例-正例的差值
            matrix_label = matrix_label.float()

            matrix_output = matrix_output - (1 - matrix_label) * 1e12
            flatten_output = matrix_output.view(-1)
            # 这里加0是因为e^0 = 1相当于在log中加了1
            flatten_output = torch.cat((torch.tensor([0]).float().to(flatten_output.device), flatten_output), dim=0)

            ctc_loss = torch.logsumexp(flatten_output * 20, dim=0)

        else:
            ctc_loss = torch.zeros(len(labels))

            for i in range(labels.shape[0]):
                if len(labels.shape) == 1:
                    label = torch.zeros(self.label_size).to(scores.device)
                    label[labels[i]] = 1
                else:
                    label = labels[i].to(scores.device)
                score = scores[i]

                # 取出负例-正例的差值
                matrix_output = score[:, None] - score[None, :]  # 这里是算出所有位置 两两之间余弦的差值
                # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
                matrix_label = label[:, None] < label[None, :]  # 取出负例-正例的差值
                matrix_label = matrix_label.float()

                matrix_output = matrix_output - (1 - matrix_label) * 1e12
                flatten_output = matrix_output.view(-1)
                # 这里加0是因为e^0 = 1相当于在log中加了1
                flatten_output = torch.cat((torch.tensor([0]).float().to(flatten_output.device), flatten_output), dim=0)

                ctc_loss[i] = torch.logsumexp(flatten_output * 20, dim=0)

        return ctc_loss
