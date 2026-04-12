# Installation

## From PyPI (once published)

```bash
pip install trntensor
pip install trntensor[neuron]   # on Neuron hardware
```

## From source

```bash
git clone git@github.com:trnsci/trntensor.git
cd trntensor
pip install -e ".[dev]"
pytest tests/ -v
```

## Hardware compatibility

NKI kernels target **Neuron SDK 2.24+** on the **Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)** AMI. Without Neuron hardware, trntensor falls through to PyTorch ops (`torch.einsum`, `torch.matmul`, `torch.bmm`) — all APIs remain functional.
