from IAT_enhance.model.IAT_main import Local_pred_S
from IAT_enhance.model.global_net import Global_pred
from IAT_enhance.model.IAT import IAT
import torch
from torch import nn


class IAT(nn.Module):
    def __init__(self, in_dim=3, with_global=True, type="lol"):
        super(IAT, self).__init__()
        # self.local_net = Local_pred()

        self.local_net = Local_pred_S(in_dim=in_dim)

        self.with_global = with_global
        if self.with_global:
            self.global_net = Global_pred(in_channels=in_dim, type=type)

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img_low):
        # print(self.with_global)
        mul, add = self.local_net(img_low)
        img_high = (img_low.mul(mul)).add(add)

        if not self.with_global:
            return mul, add, img_high

        else:
            gamma, color = self.global_net(img_low)
            b = img_high.shape[0]
            img_high = img_high.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
            img_high = torch.stack(
                [
                    self.apply_color(img_high[i, :, :, :], color[i, :, :])
                    ** gamma[i, :]
                    for i in range(b)
                ],
                dim=0,
            )
            img_high = img_high.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)
            return mul, add, img_high
