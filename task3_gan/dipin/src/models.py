"""CycleGAN networks: ResNet generator and 70x70 PatchGAN discriminator.

Follows Zhu et al. (2017), "Unpaired Image-to-Image Translation using
Cycle-Consistent Adversarial Networks", Appendix 7.2:
  generator     c7s1-64, d128, d256, R256 x 9, u128, u64, c7s1-3
  discriminator C64-C128-C256-C512 -> 1-channel patch map
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        ch = ngf
        for _ in range(2):  # downsample: d128, d256
            layers += [
                nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ch * 2),
                nn.ReLU(inplace=True),
            ]
            ch *= 2
        layers += [ResidualBlock(ch) for _ in range(n_blocks)]
        for _ in range(2):  # upsample: u128, u64
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1,
                                   output_padding=1, bias=False),
                nn.InstanceNorm2d(ch // 2),
                nn.ReLU(inplace=True),
            ]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN. Outputs a 30x30 map of real/fake logits for 256x256 input."""

    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        layers = [nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]
        ch = ndf
        for i, stride in enumerate([2, 2, 1]):  # C128, C256, C512
            layers += [
                nn.Conv2d(ch, ch * 2, 4, stride=stride, padding=1, bias=False),
                nn.InstanceNorm2d(ch * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch *= 2
        layers += [nn.Conv2d(ch, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights(net, gain=0.02):
    """N(0, 0.02) init for conv weights, as in the paper's reference code."""
    def fn(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    net.apply(fn)
    return net


def count_params(net):
    return sum(p.numel() for p in net.parameters())


def build_models(cfg):
    g, d = cfg["generator"], cfg["discriminator"]
    G_AB = init_weights(ResnetGenerator(ngf=g["ngf"], n_blocks=g["residual_blocks"]))
    G_BA = init_weights(ResnetGenerator(ngf=g["ngf"], n_blocks=g["residual_blocks"]))
    D_A = init_weights(PatchDiscriminator(ndf=d["ndf"]))
    D_B = init_weights(PatchDiscriminator(ndf=d["ndf"]))
    return G_AB, G_BA, D_A, D_B


if __name__ == "__main__":
    G, D = ResnetGenerator(), PatchDiscriminator()
    x = torch.randn(1, 3, 256, 256)
    print("G out", tuple(G(x).shape), "params", count_params(G))
    print("D out", tuple(D(x).shape), "params", count_params(D))
