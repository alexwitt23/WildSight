import Vue from 'vue';
import Router from 'vue-router';
import Home from '../components/Home.vue';
import Resnet50 from '../components/models/Resnet50.vue';
import SqueezeNet from '../components/models/Squeezenet.vue';
import Emotion from '../components/models/Emotion.vue';
import Yolo from '../components/models/Yolo.vue';
import Giraffe from '../components/models/Giraffe.vue';
import MNIST from '../components/models/MNIST.vue';

Vue.use(Router);

export default new Router({
  mode: 'hash',
  base: process.env.BASE_URL,
  routes: [
    {
      path: '*',
      name: 'home',
      component: Home,
    },
    {
      path: '/yolo',
      component: Yolo,
    },
    {
      path:'/giraffe',
      component: Giraffe,
    }
  ],
});
