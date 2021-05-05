<template> 
    <navbar></navbar>
      <div class="row">
         <h1 class="display-1 mt-5 text-center">Swift Parrots</h1>  
       </div>
       <div class="row justify-content-md-center">
          <div class="col-md-6">
          <p class="text-left">
            We created a dataset of close to 880 pictures of swift parrots from various sources.
            We downloaded around 400-500 pictures from Google Images and extracted frames
            from a couple different YouTube videos. All images were labelled using
            <a href="https://www.makesense.ai/"><b>MakeSense</b></a>.
          </p>
          <p class="text-left">
            To make our model more robust against other bird species, the parrot dataset was
            combined with <a href="https://www.kaggle.com/gpiosenka/100-bird-species"><b>this</b></a>
            dataset of 265 other bird species and about 37k images.
          </p>

          <p class="text-left">
            Having close to 900 images is a good start, but we could still use more to
            improve model performance. The parrots can be in many different environments:
            flying, bathing, eating, drinking, etc. More images of these actions will help
            the model generalize.
          </p>        
          <p class="text-left">
            Below, you can drop multiple files. Currently, the last file to be uploaded is
            visualized. After running the model, the results come back as image name,
            object class, detection confidence, x0, y0, x1, y1, where (x0, y0) and (x1, y1)
            are the top-left and bottom-right coordinates of the predicted bounding box. These
            can be downloaded as a csv file.
          </p>
       </div>
        <h4 class="text-center loading" v-if="!isModelReady && !initFailMessage">Loading model...</h4>
        <div class="spinner-border text-success" role="status" v-if='!isModelReady && !initFailMessage'></div>
      </div>
      <div class="row">
         <h3 v-if="initFailMessage">Failed to init stream and/or model - {{ initFailMessage }}</h3>
       </div>
       <div class="row mt-5 justify-content-md-center">
         <div class="col-xs-6 col-xs-offset-3">
          <input name="file" v-if="isModelReady" type="file" multiple accept="image/*" @change="uploadImage($event)" id="file-input">
          <canvas ref="canvas" class="mt-5 "></canvas>
         </div>
        <div class="col-5 mt-5 mb-5" v-if="isResultReady">
          <Slider v-model="userConfidence" @update="renderPredictionBoxes()"/>
          <p class="text-center mt-3">
            Model Confidence
          </p>
        </div>
        <div id="#results" v-if="isResultReady" class="mt-5 pb-5">
          <button v-on:click="downloadResults()" class="button btn">Download Results</button>
        </div>
      </div>
      <div class="container">
        <div class="row justify-content-center mb-5">
          <div class="col-md-6">
            <h2>Example Images</h2>
            <img v-for="image in exampleImages" v-bind:key="image" :src="image.url" class="img-fluid pb-4">
          </div>
        </div>
      </div>       
</template>

<script>

import Slider from '@vueform/slider'
import * as tf from '@tensorflow/tfjs'
import { RetinaNetDecoder } from '../../utils/retinanet_decoder_parrot'
import Navbar from '../layouts/NavBar'
const MODEL_URLS = {
  'local': 'http://localhost:8081/public/2021-04-29T17.09.05/model.json',
  'remote': 'https://cdn.jsdelivr.net/gh/alexwitt23/wildsight-models@main/2021-04-29T17.09.05/model.json'
}
let model


export default {
  name: 'app',
  el: '#results',
  components: {
    Slider,
    'navbar': Navbar
  },
  data () {
    return {
      results: {},
      userConfidence: 0.50,
      // store the promises of initialization
      image: null,
      // control the UI visibilities
      isModelReady: false,
      isResultReady: false,
      initFailMessage: '',
      // tfjs model related
      model: null,
      resultWidth: 0,
      resultHeight: 0,
      maxCanvasHeight: 800,
      maxCanvasWidth: 1200,
      filenames: [],
      modelSize: 640,
      time: 0,
      exampleImages: [
        {"url": "https://user-images.githubusercontent.com/31543169/116627198-45c55d80-a912-11eb-83ee-18f57cc8f8ce.jpg"},
        {"url": "https://user-images.githubusercontent.com/31543169/116627235-57a70080-a912-11eb-8239-4783b6c052c9.jpg"},
        {"url": "https://user-images.githubusercontent.com/31543169/116627224-5249b600-a912-11eb-84fc-d47b65998bce.jpeg"},
      ]
    }
  },
  methods: {
    uploadImage(event) {

      // initialize variables for csv
      this.csvInit()

      // iterate through input files
      this.filenames = [];
      for (let fl of event.target.files) {

        this.filenames.push(fl.name);
       
        let reader = new FileReader();
        reader.onload = e => {
          let img = document.createElement('img');
          
          img.src = e.target.result;
          img.onload = () => {
            let aspectRatio = img.width / img.height
            if (img.width > this.maxCanvasWidth && aspectRatio >= 1) {
              this.imgWidth = this.maxCanvasWidth
              this.imgHeight = this.imgWidth / aspectRatio
            }
            else if (img.height > this.maxCanvasHeight) {
              this.imgHeight = this.maxCanvasHeight
              this.imgWidth = this.imgHeight / aspectRatio
            }
            else{
              this.imgWidth = img.width
              this.imgHeight = img.height
            }
            
            this.predict(img, fl.name)
          };
        }
        reader.readAsDataURL(fl);
      }
      
    },

    async initializeBackend() {
      tf.ready().then(() => {console.log(tf.getBackend())})
    },

    async loadCustomModel () {
      let modelFilepath = process.env.NODE_ENV === 'production' ? MODEL_URLS["remote"] : MODEL_URLS["local"];
      model = await tf.loadGraphModel(modelFilepath)

      // Warmup the model first and run all the overhead-heavy intialization operations.
      const zeros = tf.zeros([1, 3, this.modelSize, this.modelSize])
      const predictions = await model.executeAsync(zeros)
      const regressions = predictions.slice([0, 0], [-1, 4])
      const class_logits = predictions.slice([0, 4], [-1, -1])
      await this.decoder.get_boxes(class_logits, regressions)

      console.log("Loaded model.")
      this.isModelReady = true
    },

    async predict(imgElement, imageName){
      this.imgElement = imgElement
      const img = tf.browser.fromPixels(imgElement).resizeBilinear([this.modelSize, this.modelSize]).toFloat().expandDims(0).transpose([0, 3, 1, 2])
      const mean = tf.tensor([0.485, 0.456, 0.406]).expandDims(0).expandDims(-1).expandDims(-1).mul(255.0)
      const std = tf.tensor([0.229, 0.224, 0.225]).expandDims(0).expandDims(-1).expandDims(-1).mul(255.0)
      const normalized = img.sub(mean).div(std)

      //timing variable
      var start = new Date();
      var predictions = await model.executeAsync(normalized)
      //timing variable
      var end   = new Date();
      var difference = new Date();
      difference.setTime(end.getTime() - start.getTime());
      this.time += difference.getMilliseconds(); 

      const regressions = predictions.slice([0, 0], [-1, 4])
      const class_logits = predictions.slice([0, 4], [-1, -1])
      const [classes, bboxes, confidences] = await this.decoder.get_boxes(class_logits, regressions)
      this.classes = classes
      this.bboxes = bboxes
      this.confidences = confidences

      this.results[imageName] = {
        "classes": classes.arraySync(),
        "bboxes": bboxes.arraySync(),
        "confidences": confidences.arraySync(),
      }

      this.renderPredictionBoxes()
      img.dispose()
      this.isResultReady = true
    return [confidences, bboxes]
    },

    async renderPredictionBoxes () {
      let cvn = this.$refs.canvas;
      cvn.width = this.imgWidth;
      cvn.height = this.imgHeight;
      let ctx = cvn.getContext("2d");  
      ctx.drawImage(this.imgElement, 0, 0, this.imgWidth, this.imgHeight);
      
      for (var i = 0; i < this.bboxes.shape[0]; i++){

        let con = this.confidences.slice([i], [1]).toFloat().dataSync()
        let arr = this.bboxes.slice([i, 0], [1, -1]).toInt().dataSync()

        const minX = arr[0] / this.modelSize * this.imgWidth
        const minY = arr[1] / this.modelSize * this.imgHeight
        const maxX = arr[2] / this.modelSize * this.imgWidth
        const maxY = arr[3] / this.modelSize * this.imgHeight

        if (con > this.userConfidence / 100) {
          ctx.beginPath()
          ctx.rect(minX, minY, maxX - minX, maxY - minY)
          ctx.lineWidth = 3
          ctx.strokeStyle = 'red'
          ctx.fillStyle = 'red'
          ctx.stroke()
          ctx.shadowColor = 'white'
          ctx.shadowBlur = 10
          ctx.font = '14px Arial bold'
        }
      }
    },

    csvInit(){
      this.header = ["image", "class", "confidence", "x0", "y0", "x1", "y1"].join(',');
      this.csv    = ''
      this.imageIdx    = 0
    },

    csvExport() {
      for (var i = 0; i < this.filenames.length; i++){
        let fileName = this.filenames[i]

        const results = this.results[fileName]
        let arrs = results["bboxes"]
        let cons = results["confidences"]

        for (var j = 0; j < arrs.length; j++){
          let con = cons[j]
          let arr = arrs[j]
          if (con > this.userConfidence / 100) {
            var row = [
              this.filenames[i],
              "swift-parrot",
              con,
              arr[0] / this.modelSize * this.imgWidth,
              arr[1] / this.modelSize * this.imgHeight,
              arr[2] / this.modelSize * this.imgWidth,
              arr[3] / this.modelSize * this.imgHeight
            ];
            this.csv += row.join(',');
            this.csv += "\n"
          }
        }
        this.imageIdx++
      }

    },
    downloadResults () {
      this.csvInit()
      this.csvExport()
      // print average processing time to console   
      console.log('Average processing time: ' + String(this.time/this.inm) + ' milliseconds' )
      
      var out = this.header + "\n" + this.csv

      const anchor = document.createElement('a');
      anchor.href = 'data:text/csv;charset=utf0-8,' + encodeURIComponent(out);
      anchor.target = '_blank';
      anchor.download = 'wildlife_results.csv';   // we should make the naming more sophisticated 
      anchor.click();
    },
  },
  mounted () {
    this.initializeBackend()
    this.decoder = new RetinaNetDecoder(1)
    this.loadCustomModel()
  }
}
</script>

<style src="@vueform/slider/themes/default.css"></style>

<style lang="scss">
.resultFrame {
  display: grid;
  video {
    grid-area: 1 / 1 / 2 / 2;
  }
  canvas {
    grid-area: 1 / 1 / 2 / 2;
  }
}
.canvas-overlay {
  pointer-events: none;
  width: 50%;
  height: 50%
}

.canvas-wrapper {
  position: relative;
}

</style>
