all:
	@echo "Usage: make [train|retrain]"

train:
	python3 -u train.py

retrain:
	rm -rf tiny_imagenet_model
	python3 train.py tiny-imagenet-200

tensorboard:
	@echo "Starting tensorboard in background... (use \"tmux a\" to attach)"
	@tmux new -d 'tensorboard --logdir=logs'
	@tmux ls

monitor:
	# @watch "iostat && nvidia-smi"
	@glances
