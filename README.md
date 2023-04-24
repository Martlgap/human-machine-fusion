


# Platform
We used the following platform to run the code:
- Ubuntu 20.04
- Python 3.8.10
- NVIDIA GeForce GTX 1070
- Cuda 11.2

# Requirements
Create a virtual environment and install the requirements:

```bash
pip install -r requirements.txt
```

You then need to add the following lines of code to the __init__.py file in the vit_pytorch site-package folder in your virtual environment:
```python
from vit_pytorch.vit_face import ViT_face
from vit_pytorch.vits_face import ViTs_face
```

Please also add the files from the [vit_pytorch_files/](vit_pytorch_files) folder to the vit_pytorch site-package folder in your virtual environment.


# Download Datasets & Models
```bash
python from utils.helper import * ;download_models(); download_datasets(); extract_datasets()
```

# Issues
- [ ] Memory of mxnet cannot be cleared so you might need to run model inference