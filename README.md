# WaveGlow 

A PyTorch implementation of the [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002)

## Model Description
WaveGlow is a flow-based generative network for speech synthesis. It is capable of generating high-quality speech from mel-spectrograms.

## Pre-trained Models
[Download Pre-trained Model](https://drive.google.com/file/d/1sz-7sJnNWnx3ttM46FyYr1GYPG4N2Q07/view?usp=sharing)

## Model & Audio Parameters
The model is configured with the following parameters (based on `waveglow_params.json`):

## Quick Start:

1. Install requirements:
```
pip install -r requirements.txt
```

2. Dataset:
The model was trained on the [Ruslan dataset](https://ruslan-corpus.github.io).

> **IMPORTANT**
>
> The audio files were resampled to 22050 Hz.

3. Features & Data Preparation:
The model uses original spectrograms derived from the dataset audio. The `prepare_data.py` script was used to generate these spectrograms for training.

4. Training with default hyperparams:
```
python train.py
```

5. Synthesize from model:
```
python generate.py --checkpoint=/path/to/model --local_condition_file=/path/to/local_conditon
```

