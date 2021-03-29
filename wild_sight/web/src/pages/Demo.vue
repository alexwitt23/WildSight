<template>
  <main-layout>
    <h3 v-if="!isModelReady && !initFailMessage">loading model ...</h3>
    <h3 v-if="initFailMessage">Failed to init stream and/or model - {{ initFailMessage }}</h3>
    <input v-if="isModelReady" type="file" accept="image/*" @change="uploadImage($event)" id="file-input">
    <canvas ref="canvas"></canvas>
  </main-layout>
  <div id="#results" v-if="isResultReady">
    <button v-on:click="downloadResults()">Download Results</button>
  </div>
</template>

<script>

import * as tf from '@tensorflow/tfjs'
import { RetinaNetDecoder } from '../../utils/retinanet_decoder'
import MainLayout from '../layouts/Main.vue'
const MODEL_URL = 'https://storage.googleapis.com/wild-sight/web_model/model.json';
//const MODEL_URL = "file://../public/web_model/model.json"
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
    }
  },
  methods: {
    uploadImage(event) {
      let reader = new FileReader();
      reader.onload = e => {
        let img = document.createElement('img');
        img.src = e.target.result;
        img.onload = () => {
          this.predict(img)
        };
      }
      reader.readAsDataURL(event.target.files[0]);
    },

    async initializeBackend() {
      tf.ready().then(() => {console.log(tf.getBackend())})
    },

    async loadCustomModel () {
      global.fetch = require('node-fetch');
      model = await tf.loadGraphModel(MODEL_URL)
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
      const [bboxes, confidences] = await this.decoder.get_boxes(class_logits, regressions)
      this.renderPredictionBoxes(imgElement, bboxes)
      this.csvExport(confidences, bboxes)
      img.dispose()
      this.isResultReady = true
    return [confidences, bboxes]
    },

    renderPredictionBoxes (imgElement, bboxes) {
      let cvn = this.$refs.canvas;
      cvn.width = 512;
      cvn.height = 512;
      let ctx = cvn.getContext("2d");  
      ctx.drawImage(imgElement, 0 ,0, 512, 512);
      // clear the canvas
      //ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      // draw results
      //confidences = confidences.dataSync()

      for (var i = 0; i < bboxes.shape[0]; i++){
        let arr = bboxes.slice([i, 0], [1, -1]).toInt().dataSync()
        
        const minX = arr[0]
        const minY = arr[1]
        const maxX = arr[2]
        const maxY = arr[3]
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
        // function to output csv, called by predict ()
    csvExport(confidences, bboxes) {

      //add headings to csv    
      this.csv = "Image,Class"

      //add box specific headings
      for (var i = 0; i < bboxes.shape[0]; i++){

        this.csv += ','
        var corners = ["Box"+String(i)+"_Confidence", "Box"+String(i)+"_X0", "Box"+String(i)+"_Y0", "Box"+String(i)+"_X1", "Box"+String(i)+"_Y1"]
        this.csv += corners.join(',');

      }
      this.csv += "\n"

      //Input values for image and class
      this.csv += "0,0"


      //add confidence and bounding box data for each box
      for (i = 0; i < bboxes.shape[0]; i++){

        this.csv += ','      
        let arr = bboxes.slice([i, 0], [1, -1]).toFloat().dataSync();
        let con = confidences.slice([0]).toFloat().dataSync()
        var row = [ con[i],arr[1],arr[0],arr[3],arr[2] ];
        this.csv += row.join(',');

      }
      this.csv += "\n"
    },
    downloadResults () {
      const anchor = document.createElement('a');
      anchor.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(this.csv);
      anchor.target = '_blank';
      anchor.download = 'wildlife_results.csv';   // we should make the naming more sophisticated 
      anchor.click();
    },
  },
  mounted () {
    this.initializeBackend()
    this.decoder = new RetinaNetDecoder()
    this.loadCustomModel()
  }
}
</script>

<style lang="scss">
body {
  margin: 0;
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
</style>
