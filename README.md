# Transformer Implementation

This repository contains an implementation of the Transformer architecture as described in the paper "Attention Is All You Need". The implementation is focused on translating sentences from one language to another. In this case, from English to Urdu.

## Project Structure

- [`config.py`](config.py): Configuration settings for the transformer model
- [`dataset.py`](dataset.py): Dataset handling and preprocessing utilities
- [`model.py`](model.py): Implementation of the transformer architecture
- [`train.py`](train.py): Training loop and optimization code
- [`test.json`](test.json): Test dataset in JSON format
- [`train.json`](train.json): Training dataset in JSON format

## Getting Started

### Prerequisites

- Python 3.6 or higher
- PyTorch

### Installation

```bash
git clone https://github.com/HAPPYLAMMA2001/Transformer_Implementation.git
cd Transformer_Implementation
pip install -r requirements.txt
```

### Usage

1. Configure the model parameters in [`config.py`](config.py)
2. Train the model:
   ```bash
   python train.py
   ```
3. Use the trained model for inference.

## Model Architecture

This implementation includes the standard transformer architecture with:
- Multi-head attention mechanism
- Position-wise feed-forward networks
- Positional encoding
- Layer normalization

## Datasets

The model can be trained on custom datasets provided in JSON format. Example datasets are included:
- [`train.json`](train.json): Used for training the model
- [`test.json`](test.json): Used for evaluation


## Acknowledgments

- "Attention Is All You Need" paper by Vaswani et al.