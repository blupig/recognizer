## CIFAR-10

CIFAR-10 classification (CNN)

### Environment Setup

Assuming `python3` and `virtualenv` are installed.

```bash
virtualenv venv
source venv/bin/activate

# To train on GPU, use `tensorflow-gpu` in requirements.txt (additional setup required).
pip3 install -r requirements.txt
```

### Training

Training takes less than 20 minutes on a single `NVIDIA Tesla K80` GPU (AWS `p2.xlarge` instance).

Or roughly 2-3 hours on an Intel i7-7567U CPU.

```
python3 train.py
```

It saves the model as `model.tflearn` in current directory.

### Classification

A trained model `model.tflearn` is required in `models` directory.

```
python3 classify.py test/1.jpg
```
