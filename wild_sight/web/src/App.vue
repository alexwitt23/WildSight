<template>
  <div id="app">
    <h3 v-if="!isModelReady && !initFailMessage">loading model ...</h3>
    <h3 v-if="initFailMessage">Failed to init stream and/or model - {{ initFailMessage }}</h3>
    <div class="resultFrame">
      <video ref="video" autoplay></video>
      <canvas ref="canvas" :width="resultWidth" :height="resultHeight"></canvas>
    </div>
  </div>
</template>

<script>
import * as tf from '@tensorflow/tfjs'
import { loadGraphModel } from '@tensorflow/tfjs-converter'
const MODEL_URL = 'http://localhost:8081/web_model/model.json';

export default {
  name: 'app',
  data () {
    return {
      // store the promises of initialization
      streamPromise: null,
      modelPromise: null,
      // control the UI visibilities
      isVideoStreamReady: false,
      isModelReady: false,
      initFailMessage: '',
      // tfjs model related
      model: null,
      videoRatio: 1,
      resultWidth: 0,
      resultHeight: 0
    }
  },
  methods: {
    async loadCustomModel () {
      this.isModelReady = false
      console.log(tf.getBackend());
      const model = await loadGraphModel(MODEL_URL)
      const cat = tf.ones([1, 3, 512, 512])
      const predictions = model.executeAsync(cat).then(predictions=> { 
        predictions[0] = predictions[0].reshape([-1]).sigmoid()
        console.log('classifications: ', predictions[0]);
        console.log('regressions: ', predictions[1]);
        const {values, indices} = tf.topk(predictions[0], 10, true)
        values.array().then(values => console.log(values));
        indices.array().then(indices => console.log(indices));
        for (let i = 0; i < values.size; i++){
          console.log(values)
        }
      })
      
      return predictions
    },
  },
  mounted () {
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
</style>