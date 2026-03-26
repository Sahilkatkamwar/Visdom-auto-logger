# Auto Visdom Logger

A small PyTorch prototype that automatically logs training metrics to Visdom.

This project demonstrates a class-based logger that captures:
- training loss
- validation loss
- training accuracy
- validation accuracy
- gradient norms
- per-parameter gradient statistics

It uses hook-based gradient capture and live Visdom plots on a simple MNIST CNN demo.

## Problem

Manual logging is repetitive.

When training PyTorch models, developers often have to write many explicit logging calls for every metric they want to track. This adds boilerplate, makes training code harder to maintain, and increases the chance of missing useful training statistics.

## Solution

This project shows a small automatic logging system for PyTorch training.

The logger:
- accepts metrics through a clean class interface
- captures gradients using hooks during the backward pass
- computes and logs gradient norms automatically
- sends metrics to Visdom for live visualization

## Features

- Automatic loss logging
- Gradient norm logging
- Hook-based gradient capture
- Actual Visdom integration
- Minimal user code
- MNIST demo with a small CNN
- Live plots for training and validation metrics

## Project Structure

```text
auto-visdom-logger/
├── logger/
│   ├── logger.py
│   └── hooks.py
├── examples/
│   └── train_mnist.py
├── utils/
│   └── grad_utils.py
├── README.md
└── requirements.txt