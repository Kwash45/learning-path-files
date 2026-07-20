MNIST ExecuTorch example for Alif Ensemble E8
=============================================

This directory contains support files for the Learning Path:

``TODO: add Learning Path link``

The learning path provides instructions on how to run an MNIST digit-classification model with ExecuTorch on the Alif Ensemble E8 DevKit using the Arm Ethos-U85 NPU.

This repository is intended for education and demonstration only. It does not contain production ready firmware, and it has not been hardened for safety, security, fault recovery, or long-running deployment.

Contents
--------

.. code-block:: text

   .
   ├── mnist_executorch/
   │   ├── SEGGER_RTT_Conf.h
   │   ├── executorch_runner.cpp
   │   ├── executorch_runner.h
   │   └── main.cpp
   ├── et_bundle.tar.gz
   ├── mnist_ethos_u85.pte
   ├── mnist_model.py
   ├── prepare_mnist_image.py
   ├── sample_one.pt
   └── train_mnist.py

The ``mnist_executorch/`` directory contains the application source files used by the CMSIS project.

The other files are supporting artifacts:

* ``mnist_ethos_u85.pte``: pre-exported ExecuTorch model for Ethos-U85.
* ``et_bundle.tar.gz``: ExecuTorch headers and static libraries for the firmware build.
* ``mnist_model.py``: PyTorch model definition used during export.
* ``train_mnist.py``: script for training the MNIST model.
* ``sample_one.pt``: representative calibration input used during quantized export.
* ``prepare_mnist_image.py``: helper script for converting a test digit image into a C header.

Usage
-----

Follow the Learning Path instructions to copy these files into the Alif VS Code template project, build the CMSIS project, flash the Alif Ensemble E8 DevKit, and view inference output with SEGGER RTT.

These files are not intended to be used as a standalone application without the Learning Path context.

License
-------

This example is provided under the BSD-3-Clause-Clear license.

See ``LICENSE.md`` for details.
