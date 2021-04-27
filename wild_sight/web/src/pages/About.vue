<template>
  <navbar></navbar>
  <div class="container-fluid">
    <div class="cover2">
        <h1 class="display-1 mt-5 mb-4 text-center">About Us</h1>
    </div>
    <div class="container">
      <div class="row justify-content-center mb-5">
        <div class="col-md-8">
          <p class="text-left">
            This application was created for a senior design project at UT Austin. The application is
            for educational and research purposes only. The entire application is open-source and can
            be found <a href="https://github.com/alexwitt23/WildSight"><b>here</b></a>. This includes
            not only the web application, but also a pipeline for training models.
          </p>
          <p class="text-left">
            We created this application to deliver a fast and reliable machine learning
            application through an accessible platform. Instead of downloading external software or pluggin
            into a cloud platform, this application runs entirely in <b>your browser</b>.
          </p>
          <h2>Animals</h2>
          <p class="text-left">
            A few different animals were selected as test classes. We chose <b>zebras</b>,
            <b>giraffes</b>, and  <b>whale sharks</b>. The main limiting factor was the available data.
            We need datasets with "bounding box" labels around these animals of interest to train our
            object detector. Object detectors are more specialized than typical classifiers because the
            former can find multiple instances of a target in an image as the latter can only assign a
            single class to an image.
          </p>
          <h2>Datasets</h2>
          <p class="text-left">
            With the specification outlined above, we found a few datasets to train a robust model:
          </p>
          <div class="list-group mb-4">
            <a href="http://lila.science/datasets/great-zebra-giraffe-id"
            class="list-group-item list-group-item-dark" aria-current="true">
              <div class="text-center">
                Great Zebra and Giraffe Count and ID (Zebras and Giraffes)
              </div>
            </a>
            <a href="http://lila.science/datasets/whale-shark-id"
            class="list-group-item list-group-item-dark" aria-current="true">
              <div class="text-center">
                Whale Shark ID (Whale sharks)
              </div>
            </a>
            <a href="https://www.kaggle.com/larusso94/shark-species"
            class="list-group-item list-group-item-dark" aria-current="true">
              <div class="text-center">
                Shark Species (Shark species other than whale shark)
              </div>
            </a>
            <a href="https://cocodataset.org/#home"
            class="list-group-item list-group-item-dark" aria-current="true">
              <div class="text-center">
                COCO Dataset (Giraffes, Zebras, and other objects)
              </div>
            </a>
          </div>  
          <p class="text-left">
            These datasets were used because they offered clean labels of the target animals. In addition
            to having images of zebras and giraffes, the COCO dataset also contains 88 other categories of
            objects. Exposing our model to what <em>isn't</em> one of our animals helps it learn what truely
            <em>is</em> an animal of interest. Such images aren't stricly required for training a model on your
            target animal unless you want the model to be robust to extraneous detections.
          </p> 
          <h2>Training</h2>
          <p class="text-left">
            A RetinaNet-34 model was trained over all the datasets (~100k images) for around 15 epochs.
            A 2080Ti and 1080Ti were used in parallel with 128 GiB of RAM. With more time and computational
            power, we'd like to explore more models, but we are limited by our pedestrian equipment. We used
            <a href="https://pytorch.org/"><b>PyTorch</b></a> to build our training pipeline.
          </p>    
          <h2>Web Application</h2>
          <p class="text-left">
            After training the model in PyTorch, its weights are converted to a format compatible with
            <a href="https://www.tensorflow.org/js"><b>TensorFlow.js</b></a>. A web UI is built with
            <a href="https://v3.vuejs.org/"><b>Vue.js</b></a> and hosted on
            <a href="https://www.heroku.com"><b>Heroku</b></a>.
          </p>
          <h2>Future Work</h2>
          <p class="text-left">
            After May 2021, constant updates to this project are unlikely. However, we'd enjoy to
            hear from anyone who would like to use this website or the codebase. Please open an
            issue on Github or email awitt23@utexas.edu if you'd like to ask any questions.
          </p>    
        </div>
      </div>
    </div>
    </div>
</template>

<script>
import NavBar from '../layouts/NavBar'
export default {
  components: {
    'navbar': NavBar
  },
  data() {
    return {
      activeColor: 'lightgreen'
    }
  }
}
</script>

<style>
body, html {
  padding: 0;
  margin: 0;
  width: 100%;
  min-height: 100vh;
}
body {
  background-color: rgb(235, 235, 235);
}
h2{
  text-align: center;
}
p {
  color: rgb(0, 0, 0);
}
</style>