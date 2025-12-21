# SCNet

This repository is the official implementation of [SCNet: Sparse Compression Network for Music Source Separation](https://arxiv.org/abs/2401.13276)

![architecture](images/SCNet.png)

---

## Installing

We recommend using [uv](https://github.com/astral-sh/uv) to create an isolated environment and install dependencies.

Install uv:

- macOS: `brew install uv`
- Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
uv sync
```

We use the accelerate package from Hugging Face for multi-gpu training.

```bash
uv run accelerate config
```

You need to modify the dataset path in the /conf/config.yaml. The dataset folder should contain the train and valid parts.

```bash
data:
  wav: /path/to/dataset
```

---

## Training

The training command is as follows. If you do not specify a path, the default path will be used.

```bash
uv run accelerate launch -m scnet.train --config_path path/to/config.yaml --save_path path/to/save/checkpoint/
```

---

## Inference

The model checkpoint was trained on the MUSDB dataset. You can download it from the following link:

[Download Model Checkpoint](https://drive.google.com/file/d/1CdEIIqsoRfHn1SJ7rccPfyYioW3BlXcW/view?usp=sharing)

The large version is now available.

- [SCNet-large](https://drive.google.com/file/d/1s7QvQwn8ag9oVstGDBQ6KZvacJkvyK7t/view?usp=drivesdk)
- [config.yaml](https://drive.google.com/file/d/1qxK7SZx6-Gsp1s3wCrj98X7--UcI4O3K/view?usp=drive_link)

We have performed normalization on the model's input during training, which helps in stabilizing the training process (no code modifications are needed during inference).

```bash
uv run python -m scnet.inference --input_dir path/to/test/dir --output_dir path/to/save/result/ --checkpoint_path path/to/checkpoint.th
```

---

## Linux (CUDA) setup

If you're on Linux with an NVIDIA GPU, PyTorch (installed via `uv sync`) will use CUDA when the NVIDIA driver is available.

1) Make sure the NVIDIA driver is installed and `nvidia-smi` works.

2) Verify CUDA is detected:

```bash
uv run python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

---

## Citing

If you find our work useful in your research, please consider citing:

```bibtex
@misc{tong2024scnet,
      title={SCNet: Sparse Compression Network for Music Source Separation}, 
      author={Weinan Tong and Jiaxu Zhu and Jun Chen and Shiyin Kang and Tao Jiang and Yang Li and Zhiyong Wu and Helen Meng},
      year={2024},
      eprint={2401.13276},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
