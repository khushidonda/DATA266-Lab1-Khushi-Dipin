"""Task 3: CycleGAN generator/discriminator architectures for Khushi's
locked configuration (ResNet-9 generator, 70x70 PatchGAN discriminator,
InstanceNorm throughout). All architecture values below are fixed by the
locked config in task3_config.json -- nothing here is a new design choice.

No prebuilt GAN/CycleGAN library module is used: every layer is composed
from plain torch.nn building blocks (Conv2d, ConvTranspose2d, InstanceNorm2d).
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Standard CycleGAN residual block:
    [ReflectionPad, Conv3x3, InstanceNorm, ReLU, ReflectionPad, Conv3x3, InstanceNorm] + skip.
    No activation after the final addition (matches the original paper)."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=True),
            nn.InstanceNorm2d(channels, affine=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=True),
            nn.InstanceNorm2d(channels, affine=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """ResNet-9 CycleGAN generator.

    c7s1-64 -> d128 -> d256 -> 9x ResidualBlock(256) -> u128 -> u64 -> c7s1-3(Tanh)

    Reflection padding throughout the c7s1 and residual-block convolutions;
    2 stride-2 convolutions for downsampling; 2 stride-2 ConvTranspose2d
    layers for upsampling; InstanceNorm after every conv (bias kept in
    conv layers immediately followed by InstanceNorm, per the standard
    CycleGAN convention); Tanh output activation so the output range is
    [-1, 1], matching the [-1, 1]-normalized input convention.
    """

    def __init__(self, in_channels=3, out_channels=3, base_filters=64, num_residual_blocks=9):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_filters, kernel_size=7, bias=True),
            nn.InstanceNorm2d(base_filters, affine=False),
            nn.ReLU(inplace=True),
        ]

        # 2 stride-2 downsampling convolutions: 64 -> 128 -> 256 channels.
        channels = base_filters
        for _ in range(2):
            layers += [
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(channels * 2, affine=False),
                nn.ReLU(inplace=True),
            ]
            channels *= 2

        # 9 residual blocks at the bottleneck resolution (256 channels).
        for _ in range(num_residual_blocks):
            layers += [ResidualBlock(channels)]

        # 2 stride-2 upsampling ConvTranspose2d layers: 256 -> 128 -> 64 channels.
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(channels, channels // 2, kernel_size=3, stride=2,
                                    padding=1, output_padding=1, bias=True),
                nn.InstanceNorm2d(channels // 2, affine=False),
                nn.ReLU(inplace=True),
            ]
            channels //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_channels, kernel_size=7, bias=True),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator: 4 conv layers (stride 2, 2, 2, 1) plus a
    final conv to a single-channel score map. InstanceNorm after the 2nd,
    3rd, and 4th conv layers (not the first, matching the standard
    implementation). Raw linear output -- no sigmoid -- since this is used
    with LSGAN (MSE loss on raw scores), not a BCE-based objective."""

    def __init__(self, in_channels=3, base_filters=64):
        super().__init__()

        def conv_block(in_ch, out_ch, stride, use_norm):
            # bias=True unconditionally: with InstanceNorm as the chosen norm
            # type, the standard/official CycleGAN convention keeps conv bias
            # enabled on every layer (use_bias = norm_layer == InstanceNorm2d,
            # a single global choice, not a per-layer "is norm applied here"
            # check). Fixed from the static audit -- the previous
            # `bias=not use_norm` disabled bias exactly on the InstanceNorm
            # layers, backwards from convention (verified numerically inert
            # here since InstanceNorm(affine=False) mean-centers any constant
            # bias away regardless, but fixed for consistency/correctness).
            block = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=True)]
            if use_norm:
                block.append(nn.InstanceNorm2d(out_ch, affine=False))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        layers = []
        layers += conv_block(in_channels, base_filters, stride=2, use_norm=False)
        layers += conv_block(base_filters, base_filters * 2, stride=2, use_norm=True)
        layers += conv_block(base_filters * 2, base_filters * 4, stride=2, use_norm=True)
        layers += conv_block(base_filters * 4, base_filters * 8, stride=1, use_norm=True)
        layers += [nn.Conv2d(base_filters * 8, 1, kernel_size=4, stride=1, padding=1, bias=True)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights(module, gain=0.02):
    """Standard CycleGAN weight initialization: Conv/ConvTranspose weights
    ~ N(0, gain^2); InstanceNorm affine params (when present) left at
    default. Applied via model.apply(init_weights)."""
    classname = module.__class__.__name__
    if hasattr(module, "weight") and ("Conv" in classname):
        nn.init.normal_(module.weight.data, mean=0.0, std=gain)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm2d" in classname and getattr(module, "affine", False):
        nn.init.normal_(module.weight.data, mean=1.0, std=gain)
        nn.init.constant_(module.bias.data, 0.0)


def build_generator():
    model = ResNetGenerator(in_channels=3, out_channels=3, base_filters=64, num_residual_blocks=9)
    model.apply(init_weights)
    return model


def build_discriminator():
    model = PatchDiscriminator(in_channels=3, base_filters=64)
    model.apply(init_weights)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def set_requires_grad(models, requires_grad):
    """Toggle requires_grad on every parameter of the given model(s), without
    using torch.no_grad() -- forward passes through a frozen discriminator
    still build a graph connecting back to whatever fed it (e.g. a
    generator's output), so gradients still flow into the generator; only
    the discriminator's own parameters stop accumulating a .grad."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        for p in model.parameters():
            p.requires_grad = requires_grad


if __name__ == "__main__":
    # Lightweight, non-training sanity checks only.
    torch.manual_seed(42)

    g_a2b = build_generator()
    g_b2a = build_generator()
    d_a = build_discriminator()
    d_b = build_discriminator()

    x = torch.randn(1, 3, 256, 256)

    g_out = g_a2b(x)
    assert g_out.shape == (1, 3, 256, 256), f"Expected generator output (1,3,256,256), got {tuple(g_out.shape)}"
    assert g_out.min().item() >= -1.0 - 1e-4 and g_out.max().item() <= 1.0 + 1e-4, (
        f"Generator output out of Tanh range [-1,1]: min={g_out.min().item()}, max={g_out.max().item()}"
    )
    print(f"[PASS] Generator: input {tuple(x.shape)} -> output {tuple(g_out.shape)}, "
          f"range=[{g_out.min().item():.4f}, {g_out.max().item():.4f}] (within [-1,1])")

    d_out = d_a(x)
    print(f"[PASS] Discriminator: input {tuple(x.shape)} -> PatchGAN score map {tuple(d_out.shape)}, "
          f"no sigmoid (raw range=[{d_out.min().item():.4f}, {d_out.max().item():.4f}])")

    n_g_a2b = count_parameters(g_a2b)
    n_g_b2a = count_parameters(g_b2a)
    n_d_a = count_parameters(d_a)
    n_d_b = count_parameters(d_b)
    total = n_g_a2b + n_g_b2a + n_d_a + n_d_b

    print(f"\nParameter counts:")
    print(f"  G_A2B: {n_g_a2b:,}")
    print(f"  G_B2A: {n_g_b2a:,}")
    print(f"  D_A:   {n_d_a:,}")
    print(f"  D_B:   {n_d_b:,}")
    print(f"  Total: {total:,}")
