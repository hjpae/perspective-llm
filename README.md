# perspective-llm

## instance setup 

Generate the SSH key:  

```bash
git config --global user.name "hjpae"
git config --global user.email "hnjpae@gmail.com"

ssh-keygen -t ed25519 -C "hnjpae@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Add SSH key and then authorize connection to GitHub:  

```bash
ssh -T git@github.com
```

Clone the repository:  

```bash
git clone git@github.com:hjpae/perspective-llm.git
cd perspective-llm
git remote -v
```

If overwriting the existing repo, use this:  

```bash
git remote remove origin
git remote add origin git@github.com:hjpae/perspective-llm.git
```


## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate perspective-llm
```

in case, to match spyder kernels version:

```bash
conda install -c conda-forge "spyder-kernels>=x.y,<x.z"
```

check the python version: 

```bash
python --version
python -c "import sys; print(sys.executable)"
```

Install PyTorch separately if needed:  

```bash
pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Verify the installation with:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

---
