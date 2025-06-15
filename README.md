# GSlang: A 3DGS Implementation in Slang

## Overview

GSlang is a 3D Gaussian Splatting implementation using the Slang shading language. This project leverages slangpy to create a high-performance 3D Gaussian rendering system.

## Setup

### Prerequisites

- Python 3.12 or newer
- [uv](https://github.com/astral-sh/uv) - A fast Python package installer and resolver

### Installation with uv

[uv](https://github.com/astral-sh/uv) is a fast, reliable Python package installer and resolver. Follow these steps to set up the project:

1. **Install uv** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Clone the repository**:
   ```bash
   git clone git@github.com:fangjunzhou/gslang.git
   cd gslang
   ```

3. **Initialize submodules**:
   ```bash
   git submodule update --init --recursive
   ```

4. **Create a virtual environment and install dependencies**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync
   ```

### Using Nix (Alternative)

If you prefer using Nix:

```bash
nix develop
```

This will set up a development environment with all required dependencies.

## Running Tests

```bash
pytest tests
```

## Usage

Basic example of how to use the viewer:

```bash
python -m scripts.gs_viewer your_model.ply
```