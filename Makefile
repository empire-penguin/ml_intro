SETUP:
	pip3 install -r requirements.txt

FETCHDATA: 
	mkdir data
	cd data
	curl https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz | tar xzvf -

MAIN: main.py
	python main.py task-1-default-config

START: caption_utils.py coco_dataset.py dataset_factory.py experiment.py file_utils.py
	chmod +x script.sh
	./script.sh