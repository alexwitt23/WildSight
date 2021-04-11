<template> 
    <main-layout>
      <div class="row">
         <h1 class="display-1 mt-5 text-center">Demo</h1>  
       </div>
       <div class="row">
         <div class="col">
           <div closs="col-5">
            <p class="text-center">
             The model's power is showcased here. Grab some pictures of a zebra, giraffe, or
             whale shark.
            </p>
            </div>
            <a href="https://github.com/alexwitt23/WildSight/releases/download/v0.0.1/model.json">Download model</a>
           <div class="spinner-border text-success text-center m-auto d-block mt-5" role="status" v-if='!isModelReady && !initFailMessage'>
            <span class="visually-hidden">Loading model...</span>
           </div>
          <h4 class="text-center loading" v-if="!isModelReady && !initFailMessage">Loading model...</h4>
        </div>
      </div>
      <div class="row">
         <h3 v-if="initFailMessage">Failed to init stream and/or model - {{ initFailMessage }}</h3>
       </div>
       <div class="row mt-5">
         <div class="col-xs-6 col-xs-offset-3">
          <input name="file" v-if="isModelReady" type="file" multiple accept="image/*" @change="uploadImage($event)" id="file-input">
           <canvas ref="canvas" class="mt-5 pb-5"></canvas>
        </div>
      </div>
    </main-layout>
       
    <div id="#results" v-if="isResultReady" class="mt-5 pb-5">
      <button v-on:click="downloadResults()" class="button btn">Download Results</button>
    </div>
</template>

<script>

import * as tf from '@tensorflow/tfjs'
import { RetinaNetDecoder } from '../../utils/retinanet_decoder'
import MainLayout from '../layouts/Main.vue'
import CLASS_NAMES from "../../utils/class_names"
const MODEL_URLS = {
  //'local': 'https://storage.googleapis.com/wild-sight/2021-04-02T01.03.47/model.json',
  //'remote': 'https://github.com/alexwitt23/WildSight/releases/download/v0.0.1/model.json',
  //'local': 'https://github.com/alexwitt23/WildSight/releases/download/v0.0.1/model.json',
  'local': 'https://github.com/alexwitt23/WildSight/releases/download/v0.0.1/model.json',
  'remote': 'https://github.com/alexwitt23/WildSight/releases/download/v0.0.1/model.json'
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
      maxCanvasWidth: 1200,
      filenames: [],
      time: 0
    }
  },
  methods: {
    uploadImage(event) {

      // initialize variables for csv
      this.csvInit()

      // iterate through input files
      for (let fl of event.target.files) {

        this.filenames.push( fl.name );
       
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

      var result = await fetch(modelFilepath, {
        headers: new Headers(
          {
            'Access-Control-Allow-Origin': 'http://localhost:8081/',
            'Access-Control-Allow-Headers': 'http://localhost:8081/',
            'Access-Control-Allow-Methods': "X-Requested-With, Content-Type, Authorization",
            'Accept': 'application/octet-stream',
            'User-Agent': 'request module',
        }), 
        "mode": "no-cors"
      })
      console.log(result)
      model = await tf.loadGraphModel(modelFilepath, {requestInit: {
        headers: new Headers(
          {
            'Access-Control-Allow-Origin': "*",
            'Access-Control-Allow-Headers': "X-Requested-With, Content-Type, Authorization",
            'Access-Control-Allow-Methods': "GET, POST, PUT, DELETE, PATCH, OPTIONS",
        }), 
      }})
      this.isModelReady = true
      const zeros = tf.zeros([1, 3, 512, 512])
      const predictions = await model.executeAsync(zeros)
      const regressions = predictions.slice([0, 0], [-1, 4])
      const class_logits = predictions.slice([0, 4], [-1, -1])

      await this.decoder.get_boxes(class_logits, regressions)
      console.log("Loaded model.")
    },

    async predict(imgElement){
      //console.log(imgElement)
      const img = tf.browser.fromPixels(imgElement).resizeBilinear([512, 512]).toFloat().expandDims(0).transpose([0, 3, 1, 2])
      console.log(img.slice([0, 0, 0, 0], [-1, -1, 1, 1]).dataSync())
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
      this.renderPredictionBoxes(imgElement, bboxes)
      this.csvExport(classes, confidences, bboxes)
      img.dispose()
      this.isResultReady = true
    return [confidences, bboxes]
    },

    renderPredictionBoxes (imgElement, bboxes) {
      let cvn = this.$refs.canvas;
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
      // initilize header string
      this.header = ["image", "class", "confidence", "x0", "y0", "x1", "y1"].join(',');
      this.csv    = ''        // initialize csv string
      this.inm    = 0         // image number

    },
    // function to output csv, called by predict ()
    csvExport(classes, confidences, bboxes) {

      // add confidence and bounding box data for each box
      for (var i = 0; i < bboxes.shape[0]; i++){
    
        let arr = bboxes.slice([i, 0], [1, -1]).toFloat().dataSync();
        let con = confidences.slice([0]).toFloat().dataSync()
        let cls = classes.slice([0]).toFloat().dataSync();
        
        var row = [
          this.filenames[this.inm],
          String(CLASS_NAMES[cls[i]]),
          con[i],
          arr[0],
          arr[1],
          arr[2],
          arr[3]
        ];
        this.csv += row.join(',');
        this.csv += "\n"
      }
      this.inm++

    },
    downloadResults () {
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
    this.decoder = new RetinaNetDecoder(CLASS_NAMES.length)
    this.loadCustomModel()
  }
}
</script>

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
