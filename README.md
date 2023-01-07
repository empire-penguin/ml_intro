# ml_intro 
![GitHub release](https://img.shields.io/badge/Release-v0.1.0-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repo serves as an introduction into some modern ML paradigms and the PyTorch framework. 

## Installation
**Method 1:** Use a virtual environment to install the required packages.
```bash
python3 -m venv venv
source venv/bin/activate
```
To install the required packages, run the following command:
```bash
pip3 install -r requirements.txt
```

**Method 2:** Use Docker to run the code.
```bash
docker build -t ml_intro .
docker run -it ml_intro
```

## Usage
To run the code, run the following command:
```bash
python3 main.py <config_file>
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements
* [PyTorch](https://pytorch.org/)

