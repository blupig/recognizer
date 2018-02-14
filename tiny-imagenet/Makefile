all:
	@echo "Usage: make [train|retrain]"

load_data:
	python3 read_images.py tiny-imagenet-200

train:
	python3 train.py tiny-imagenet-200

retrain:
	rm -rf tiny_imagenet_model
	python3 train.py tiny-imagenet-200
