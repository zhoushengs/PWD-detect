

PWD-detect is a custom computer vision and object detection repository for pine wilt disease detection built on top of the [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO framework. It provides end-to-end support for model training, evaluation, inference, and containerized deployment.



## Features

- **Ultralytics Backbone:** Built directly with modern YOLO detection architectures.
- **Custom Config Support:** Ready-to-use dataset definitions via YAML configurations (e.g., `c1.yaml`).
- **Comprehensive Tooling:** Includes evaluation scripts (`test.py`), documentation setups, and automated testing workflows.

---

## Requirements

- **Python:** `>= 3.8`
- **PyTorch:** `>= 1.8` (CUDA recommended for GPU acceleration)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/zhoushengs/PWD-detect.git](https://github.com/zhoushengs/PWD-detect.git)
   cd PWD-detect
