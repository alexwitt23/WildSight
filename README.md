# WildSight
> Accelerating conservation with AI

![header](https://user-images.githubusercontent.com/31543169/111704559-565fbf80-880d-11eb-99d1-118a40bfd014.jpg)


This codebase was created for a senior design project in COE 374. This project is
currently in development.


## Setup
There are two main projects within this monorepo. One for training AI models and another
for using the models in the browser. The browser application is housed in
`wild_sight/web`.

It's recommended for this code to be run on Ubuntu. As of February 2021, Ubuntu
18.04 is the main development OS, but 20.xx likely works, too. You may also use WSL.

To get the environment setup, run:

```bash
python3 -m pip install -r requirements.txt
```

## Training

```
PYTHONPATH=. wild_sight/train/detection/train.py \
    --config wild_sight/train/configs/vovnet-det.yaml
```

## Web

Change directory into `wild_sight/web` and find the README in that folder. There you
will see the instructions for setting up the web project.
