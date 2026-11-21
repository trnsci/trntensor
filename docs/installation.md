# Installation

## From PyPI

```bash
pip install trntensor

# Trn2/Trn3 with TorchNeuron (native PyTorch, Neuron SDK 2.33+, Python ≥ 3.12)
pip install "trntensor[torchneuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com

# Trn1 / PyTorch-XLA lineage
pip install "trntensor[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

## From source

```bash
git clone git@github.com:trnsci/trntensor.git
cd trntensor
pip install -e ".[dev]"
pytest tests/ -v
```

## Hardware compatibility

The primary path is **native TorchNeuron** on Trn2/Trn3 (Neuron SDK 2.33+, `torch` 2.14, `torch-neuronx` 2.14.*), where each kernel is a `torch.ops.trntensor.*` custom op on the `neuron` device. The **PyTorch/XLA** path (Trn1, `torch-neuronx` 2.9 lineage, `nki>=0.3.0`, Neuron SDK 2.29+) is kept in maintenance. See [NKI dispatch](api/nki.md#dispatch-path) for how the path is chosen. Without Neuron hardware, trntensor falls through to PyTorch ops (`torch.einsum`, `torch.matmul`, `torch.bmm`) — all APIs remain functional.
