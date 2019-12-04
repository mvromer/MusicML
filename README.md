# MusicML
Final project for scalable machine learning class. Based on Transformer machine learning model.

## Setup
First create a virtual environment inside the root of your cloned repository.

```
python -m venv .venv
```

Then, activate your virtual Python environment. This is showing the PowerShell activation script.
There are other scripts for Bash and Windows Command Prompt.

```
.venv\Scripts\Activate.ps1
```

Next, install this repository's pip requirements.

```
pip install -r requirements.txt
```

Finally, install PyTorch. This is how to install it for Python 3.7 on Windows with CUDA support.

```
pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
```

## Scripts
The mha.py script demonstrates the multihead attention mechanism described in Vaswani et al. paper
_Attention Is All You Need_.
