This is the official repository for the paper: 

[TACKLING FACE VERIFICATION EDGE CASES: IN-DEPTH ANALYSIS AND HUMAN-MACHINE FUSION APPROACH](https://arxiv.org/pdf/2304.08134.pdf)

accepted and published at the [MVA Conference 2023, Japan](https://www.mva-org.jp/mva2023/). 

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


# Citation
If you find our work useful please consider a citation:

```bibtex
@article{knoche2023tackling,
  title={Tackling Face Verification Edge Cases: In-Depth Analysis and Human-Machine Fusion Approach},
  author={Knoche, Martin and Rigoll, Gerhard},
  journal={arXiv preprint arXiv:2304.08134},
  year={2023}
}
```
