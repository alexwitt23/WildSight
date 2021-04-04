<template>
  <main-layout></main-layout>
  <div class="container">
    <h1>Demo</h1>
    <p>
      The demo is a chance to show our models' capabilities. Below you can select which
      animals you'd like to have the model find. Grad a picture(s) you'd like to run the
      model on and checkout the results. The exact results can be downloaded in a CSV file.
    </p>
    <h3 v-if="!isModelReady && !initFailMessage">loading model ...</h3>
    <h3 v-if="initFailMessage">Failed to init stream and/or model - {{ initFailMessage }}</h3>
    <input v-if="isModelReady" type="file" multiple accept="image/*" @change="uploadImage($event)" id="file-input">
    <canvas ref="canvas"></canvas>
    <div id="#results" v-if="isResultReady">
      <button v-on:click="downloadResults()">Download Results</button>
    </div>
  </div>
</template>

<script>

import * as tf from '@tensorflow/tfjs'
import { RetinaNetDecoder } from '../../utils/retinanet_decoder'
import MainLayout from '../layouts/Main.vue'
import CLASS_NAMES from "../../utils/class_names"
const MODEL_URLS = {
  'remote': 'https://storage.googleapis.com/wild-sight/2021-04-02T01.03.47/model.json',
  'local': 'http://localhost:8081/public/2021-04-02T01.03.47/model.json'
}
let model


export default {
  name: 'app',
  el: '#results',
  components: {
    MainLayout
  },
  data () {
    return {
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
      maxCanvasWidth: 1200
    }
  },
  methods: {
    uploadImage(event) {

      // initialize variables for csv
      this.csvInit()

      // iterate through input files
      for (let fl of event.target.files) {

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
            console.log(this.imgWidth, this.imgHeight, aspectRatio)
            this.predict(img)
          };
        }
        reader.readAsDataURL(fl);
      }
    },

    async initializeBackend() {
      tf.ready().then(() => {console.log(tf.getBackend())})
    },

    async loadCustomModel () {
      console.log(process.env.NODE_ENV)
      let modelFilepath = process.env.NODE_ENV === 'production' ? MODEL_URLS["remote"] : MODEL_URLS["local"];
      model = await tf.loadGraphModel(modelFilepath)
      this.isModelReady = true
      const zeros = tf.zeros([1, 3, 512, 512])
      const predictions = await model.executeAsync(zeros)
      const regressions = predictions.slice([0, 0], [-1, 4])
      const class_logits = predictions.slice([0, 4], [-1, -1])

      await this.decoder.get_boxes(class_logits, regressions)
      console.log("Loaded model.")
    },

    async predict(imgElement){
      const img = tf.browser.fromPixels(imgElement).resizeBilinear([512, 512]).toFloat().expandDims(0).transpose([0, 3, 1, 2])
      
      const mean = tf.tensor([0.485, 0.456, 0.406]).expandDims(0).expandDims(-1).expandDims(-1).mul(255.0)
      const std = tf.tensor([0.229, 0.224, 0.225]).expandDims(0).expandDims(-1).expandDims(-1).mul(255.0)
      
      const normalized = img.sub(mean).div(std)
      var predictions = await model.executeAsync(normalized)
      const regressions = predictions.slice([0, 0], [-1, 4])
      const class_logits = predictions.slice([0, 4], [-1, -1])
      const [classes, bboxes, confidences] = await this.decoder.get_boxes(class_logits, regressions)
      this.renderPredictionBoxes(imgElement, bboxes)
      this.csvExport(classes, confidences, bboxes)
      img.dispose()
      this.isResultReady = true
    return [confidences, bboxes]
    },

    renderPredictionBoxes (imgElement, bboxes) {
      let cvn = this.$refs.canvas;
      console.log(window.innerWidth);
      cvn.width = this.imgWidth;
      cvn.height = this.imgHeight;
      let ctx = cvn.getContext("2d");  
      ctx.drawImage(imgElement, 0 ,0, this.imgWidth, this.imgHeight);

      for (var i = 0; i < bboxes.shape[0]; i++){
        let arr = bboxes.slice([i, 0], [1, -1]).toInt().dataSync()
        
        const minX = arr[0] / 512 * this.imgWidth
        const minY = arr[1] / 512 * this.imgHeight
        const maxX = arr[2] / 512 * this.imgWidth
        const maxY = arr[3] / 512 * this.imgHeight
        const score = 100
        if (score > 75) {
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
    // function to set up initial headings for csv file
    csvInit(){

      this.header = 'Image'   // initilize header string
      this.csv    = ''        // initialize csv string
      this.cls    = ''        // class designation
      this.inm    = 0         // image number
      this.mbx    = 0         // number of boxes currently listed in the header string

    },
    // function to output csv, called by predict ()
    csvExport(classes, confidences, bboxes) {
      
      //add any box specific headings not already added
      for (var i = this.mbx; i < bboxes.shape[0]; i++){

        this.header += ','
        var corners  = ["class", "confidence", "x0", "y0", "x1", "y1"]
        this.header += corners.join(',');

      }
      if (this.mbx < bboxes.shape[0]) {this.mbx = bboxes.shape[0];} 

      //Input image identifier
      this.csv += this.inm

      //add confidence and bounding box data for each box
      for (i = 0; i < bboxes.shape[0]; i++){

        this.csv += ','      
        let arr = bboxes.slice([i, 0], [1, -1]).toFloat().dataSync();
        let con = confidences.slice([0]).toFloat().dataSync()
        let cls = classes.slice([0]).toFloat().dataSync();
        
        var row = [String(CLASS_NAMES[cls[i]]), con[i], arr[1], arr[0], arr[3], arr[2] ];
        this.csv += row.join(',');

      }

      this.csv += "\n"
      this.inm++

    },
    downloadResults () {
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
    this.decoder = new RetinaNetDecoder(CLASS_NAMES.length)
    this.loadCustomModel()
  }
}
</script>

<style lang="scss">
body {
  margin: 0;
}
.container {
  .h1 {
    color: black;
  }
  text-align: center;
  align-content: center;
  .uploadBox {
    align-content: center;
  }
}
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
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  width: 100%;
  height: 100%
}

.canvas-wrapper {
  position: relative;
}

.download {
  padding-top: 50px;
}

</style>
