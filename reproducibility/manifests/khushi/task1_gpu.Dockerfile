# Task 1 (Khushi) GPU environment: the recipe used on 2026-09-25 to build the image that
# ran the smoke test, the full run task1_full_20260925_204011, the evaluation and the plots
# on the RTX 5090 lab machine. Only comments were added; FROM and RUN are as built.
#
# Why not the lab image from DockerImages/PyTorch.bat: gdevakumar/pytorch:latest
# (sha256:98a489e952cce30f86816be8111f4b800298591ee19258f7d3b54fc8c4ae1fea) ships PyTorch
# 2.1.2 + CUDA 12.1 with kernels only up to sm_90. On the RTX 5090 (sm_120)
# torch.cuda.is_available() is True but the first kernel fails with
# "no kernel image is available for execution on the device".
#
# Build (no build-context files are needed):
#   docker pull pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime
#   docker build -t khushi-task1-gpu:torch2.14-cu130 -f reproducibility/manifests/khushi/task1_gpu.Dockerfile reproducibility/manifests/khushi
#
# On 2026-09-25 the base tag resolved to
#   pytorch/pytorch@sha256:9c99fafa01edfaa3d16da8c209b38b5970bb6fd6e72725ef60efc901489f70c6
# and the build produced image id
#   sha256:f1772b5467574f78e175cbc4f67f0e74e81a87ba77d872cf97829cc1b1745ba7
# The pip install below is unpinned; the exact versions it installed are in environment.txt
# (pip freeze, identical to /opt/pip_freeze_at_build.txt). For an exact rebuild, pin the base
# with "@sha256:9c99..." and install datasets==5.0.1 matplotlib==3.11.2 (and their listed deps).
FROM pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime

# The base image installs torch system-wide with pip (PEP 668 marker present,
# no ensurepip/venv), so extra packages are added the same way. torch and
# numpy are constrained to the base image's versions so they cannot change.
RUN pip freeze | grep -E '^(torch|numpy)==' > /tmp/base_constraints.txt \
    && pip install --no-cache-dir --break-system-packages -c /tmp/base_constraints.txt datasets matplotlib \
    && pip freeze > /opt/pip_freeze_at_build.txt

WORKDIR /workspace
