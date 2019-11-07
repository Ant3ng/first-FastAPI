# first-FastAPI
Just returning predicting number

# Setup
```
$ git clone https://github.com/Ant3ng/first-FastAPI.git
$ cd first-FastAPI
$ pipenv sync
```

if you don't have pipenv
osx:
```
brew install pipenv
```

# how to run server
```
$ pipenv run uvicorn main:app --reload
```

# endpoint
go to http://127.0.0.1:8000/docs or http://127.0.0.1:8000/redoc .


# files for understanding
- dataset
    - digit_2.png: for inference
    - mnist: for training

- main.py: make prediction and return into endpoint
- training.ipynb: training based on mnist dataset
