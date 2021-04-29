<template> 
    <main-layout>
      <div class="row">
         <h1 class="display-1 mt-5 text-center">Swift Parrots</h1>  
       </div>
       <div class="row justify-content-md-center">
          <div class="col-md-6">
          <p class="text-left">
            Currently, our model can find <b>giraffes</b>, <b>zebras</b>, and <b>whale sharks</b> in
            images. We recommend using images that are 512 by 512 pixels or larger in size. If the
            images are too small, it's likely the model will not find your animal.
          </p>
          <p class="text-left">
            To get started, browse the internet or a collection of images for zebras, giraffes or whale
            sharks. You can upload multiple images at once and recieve the results back as a CSV file. The
            last image uploaded will have the results visualized. You may also upload one image at a time to
            see the model's results in the display window. The results come back as
            image name, class, confidence, x0, y0, x1, y1, where (x0, y0) and (x1, y1) are the top-left and
            bottom-right coordinates of the predicted box.  A few example images are given below.
          </p>
          <p class="text-left">
            The results come back as image name, class, confidence, x0, y0, x1, y1, where (x0, y0) and (x1, y1)
            are the top-left and bottom-right coordinates of the predicted bound.
          </p>
          <h2>Known limitations</h2>
          <p class="text-left">
            Our initial model also works much better when the input images contain just a few animals
            of interest. This means performance might be poor for animals that are positioned behind each
            other.
          </p>
          <p class="text-left">
            We've seen poor performance on images where animals are drinking from water sources and reflections in
            the water are present. This is a gap in our training data.
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
          <div class="col-md-8">
            <h2>Example Images</h2>
            <img v-for="image in exampleImages" v-bind:key="image" :src="image.url" class="img-fluid pb-4">
          </div>
        </div>
      </div>
      
    </main-layout>
       
</template>

<script>
import Slider from '@vueform/slider'
import * as tf from '@tensorflow/tfjs'
import { RetinaNetDecoder } from '../../utils/retinanet_decoder'
import MainLayout from '../layouts/Main.vue'
import CLASS_NAMES from "../../utils/class_names"
const MODEL_URLS = {
  'local': 'http://localhost:8081/public/2021-04-07T13.19.08/model.json',
  'remote': 'https://cdn.jsdelivr.net/gh/alexwitt23/wildsight-models@main/2021-04-07T13.19.08/model.json'
}
let model


export default {
  name: 'app',
  el: '#results',
  components: {
    MainLayout,
    Slider,
  },
  data () {
    return {
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
      time: 0,
      exampleImages: [
        {"url": "https://user-images.githubusercontent.com/31543169/114466579-4b9b0f00-9bae-11eb-9c05-c1dd5a874e34.jpg"},
        {"url": "https://user-images.githubusercontent.com/31543169/114466605-53f34a00-9bae-11eb-8683-12ce029339f5.jpg"},
        {"url": "https://user-images.githubusercontent.com/31543169/114466639-5d7cb200-9bae-11eb-8ec8-851c74c6d556.jpg"},
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
      this.imgElement = imgElement
      const img = tf.browser.fromPixels(imgElement).resizeBilinear([512, 512]).toFloat().expandDims(0).transpose([0, 3, 1, 2])
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

        const minX = arr[0] / 512 * this.imgWidth
        const minY = arr[1] / 512 * this.imgHeight
        const maxX = arr[2] / 512 * this.imgWidth
        const maxY = arr[3] / 512 * this.imgHeight

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
    // function to set up initial headings for csv file
    csvInit(){
      // initilize header string
      this.header = ["image", "class", "confidence", "x0", "y0", "x1", "y1"].join(',');
      this.csv    = ''        // initialize csv string
      this.inm    = 0         // image number

    },
    // function to output csv, called by predict ()
    csvExport() {
      // add confidence and bounding box data for each box
      for (var i = 0; i < this.bboxes.shape[0]; i++){
    
        let arr = this.bboxes.slice([i, 0], [1, -1]).toFloat().dataSync();
        let con = this.confidences.slice([i], [1]).toFloat().dataSync()[0]
        let cls = this.classes.slice([i], [1]).toFloat().dataSync()[0];

        if (con > this.userConfidence / 100) {
          var row = [
            this.filenames[this.inm],
            String(CLASS_NAMES[cls]),
            con,
            arr[0] / 512 * this.imgWidth,
            arr[1] / 512 * this.imgHeight,
            arr[2] / 512 * this.imgWidth,
            arr[3] / 512 * this.imgHeight
          ];
          this.csv += row.join(',');
          this.csv += "\n"
        }
      }
      this.inm++

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
    this.decoder = new RetinaNetDecoder(CLASS_NAMES.length)
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
