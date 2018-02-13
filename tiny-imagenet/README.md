## Tiny ImageNet

Tiny ImageNet classification (CNN) using TensorFlow

### Environment Setup

Assuming `python3` and `virtualenv` are installed.

```bash
virtualenv venv
source venv/bin/activate

# To train on GPU, use `tensorflow-gpu` in requirements.txt (additional setup required).
pip3 install -r requirements.txt
```

### Training

Training takes X on a single GCP [`X`](https://cloud.google.com/compute/docs/machine-types#standard_machine_types) instance with 1 [`NVIDIA Tesla K80`](http://www.nvidia.com/object/tesla-k80.html) GPU attached.

Or around X on an Intel [i7-7567U](https://ark.intel.com/products/97541/Intel-Core-i7-7567U-Processor-4M-Cache-up-to-4_00-GHz) CPU.

```
python3 train.py
```

It saves the model as `tiny_imagenet_model` in current directory.

### Classification

A trained model `tiny_imagenet_model` is required in working directory.

```
python3 classify.py test/*.png
```
