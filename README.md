MNIST ExecuTorch example for Alif Ensemble E8
=============================================

This directory contains support files for the Learning Path:

https://learn.arm.com/learning-paths/embedded-and-microcontrollers/observing-ethos-u-on-alif/

The learning path provides instructions on how to run an MNIST digit-classification model with ExecuTorch on the Alif Ensemble E8 DevKit using the Arm Ethos-U85 NPU.

This repository is intended for education and demonstration only.

## Contents

The [mnist_executorch/](mnist_executorch/) directory contains the application source files used by the CMSIS project.

The other files are supporting artifacts:

- [mnist_ethos_u85.pte](mnist_ethos_u85.pte): pre-exported ExecuTorch MNIST model for Ethos-U85.
- [et_bundle.tar.gz](et_bundle.tar.gz): compressed archive containing ExecuTorch headers and static libraries for the firmware build.
- [mnist_model.py](mnist_model.py): PyTorch model definition used during export.
- [train_mnist.py](train_mnist.py): script for training the MNIST model.
- [sample_one.pt](sample_one.pt): representative calibration input used during quantized export.
- [prepare_mnist_image.py](prepare_mnist_image.py): helper script for converting a PNG or JPEG digit image into a C header.

Usage
-----

Follow the Learning Path instructions to copy these files into the Alif VS Code template project, build the CMSIS project, flash the Alif Ensemble E8 DevKit, and view inference output with SEGGER RTT.

These files are not intended to be used as a standalone application without the Learning Path context.

License
-------

This repository uses the Arm Education End User License Agreement for teaching and learning content. See ``LICENSE.md`` for details.
