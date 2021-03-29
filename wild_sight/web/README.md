# Browser Inference Application

## Model Download

Download the trained model from [this](https://drive.google.com/file/d/1HPKMybFDaNlTEY2Az7kG9EgzA88YuC1B/view?usp=sharing)
link. Extract the folder `web_model` and place it in `wild_sight/web`.


## Project setup

All the command show here assume you are in the same folder as this README file.
You'll need to install the `http-server` package so we can acess the model
from the browser. You'll need sudo privileges, and run:

```
npm install --global http-server
```

Then, install the JS dependencies with:

```
npm install --global http-server
yarn install
```

## Development

For development purposes, run these command in two different terminals.

```
yarn serve
http-server . --cors -p=8081
```

### Compiles and minifies for production
```
yarn build
```

### Lints and fixes files
```
yarn lint
```


## Heroku
```
git push heroku main
```
