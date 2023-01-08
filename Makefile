SETUP:
	pip3 install -r requirements.txt

FETCHDATA: 
	mkdir data
	curl https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz | tar xzvf -
	mv cifar-100-python data
	rm -rf cifar-100-python

MAIN: main.py
	python3 main.py configs/default.json

START: example_dataset.py dataset_factory.py experiment.py file_utils.py
	chmod +x script.sh
	./script.sh