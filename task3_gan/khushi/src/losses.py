"""Task 3: loss functions for Khushi's locked CycleGAN configuration --
LSGAN adversarial loss, cycle-consistency L1, identity L1. Raw (unweighted)
components are returned alongside weighted totals so the training loop can
log both."""

import torch
import torch.nn as nn

_mse = nn.MSELoss()
_l1 = nn.L1Loss()


def lsgan_generator_loss(discriminator_output_on_fake, real_target=1.0):
    """G wants D(fake) to look real: MSE(D(fake), 1)."""
    target = torch.full_like(discriminator_output_on_fake, real_target)
    return _mse(discriminator_output_on_fake, target)


def lsgan_discriminator_loss(discriminator_output_on_real, discriminator_output_on_fake,
                              real_target=1.0, fake_target=0.0):
    """D wants D(real)->1 and D(fake)->0; averaged 0.5*(real_loss + fake_loss),
    matching the standard CycleGAN convention. `discriminator_output_on_fake`
    must come from a detached (or replay-pool) fake tensor -- this function
    does not detach anything itself."""
    real_labels = torch.full_like(discriminator_output_on_real, real_target)
    fake_labels = torch.full_like(discriminator_output_on_fake, fake_target)
    loss_real = _mse(discriminator_output_on_real, real_labels)
    loss_fake = _mse(discriminator_output_on_fake, fake_labels)
    total = 0.5 * (loss_real + loss_fake)
    return total, loss_real, loss_fake


def cycle_consistency_loss(real, reconstructed):
    """Raw (unweighted) L1 distance between original and reconstructed image."""
    return _l1(reconstructed, real)


def identity_loss(real, identity_output):
    """Raw (unweighted) L1 distance between a domain's real image and the
    same-domain generator's output when fed that real image directly."""
    return _l1(identity_output, real)
