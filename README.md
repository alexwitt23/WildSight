# WildSight

This codebase was created for a senior design project in COE 374.


## Setup
It's recommended for this code to be run on Ubuntu. As of February 2021, Ubuntu
18.04 is the main development OS, but 20.xx likely works, too.

To get the environment setup, run:

```bash
python3 -m pip install -r requirements.txt
```

## Training

TODO(alex)

## Flask
(Soon to be removed in favor of Vue.js)

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=-1 FLASK_APP=wild_sight/web/main.py flask run
```

## Web

```
npm install
npm run serve
```
