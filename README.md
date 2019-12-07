# MusicML
Final project for scalable machine learning class. Based on Transformer machine learning model.

## Setup
First create a virtual environment inside the root of your cloned repository.

### For Windows
```
python -m venv .venv
```

Then, activate your virtual Python environment. This is showing the PowerShell activation script.
There are other scripts for Bash and Windows Command Prompt.

```
.venv\Scripts\Activate.ps1
```

### For Mac
Python 3 already ships virtualenv, but if it’s not installed in your environment for some reason, you can install it via the package for your operating systems, otherwise you can install from pip:
```
pip install virtualenv
```
You can create and activate a virtualenv by:
```
python3 -m venv venv
. venv/bin/activate
```

Next, install this repository's pip requirements.

```
pip install -r requirements.txt
```

Finally, install PyTorch. This is how to install it for Python 3.7 on Windows with CUDA support.

```
pip install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
```
Or isntall it using Anaconda on Mac
```
conda install pytorch===1.3.1 torchvision===0.4.2 -c pytorch
```

## Scripts
The mha.py script demonstrates the multihead attention mechanism described in Vaswani et al. paper
_Attention Is All You Need_.


## Dataset
The project levages [The MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) for trainning purpose.