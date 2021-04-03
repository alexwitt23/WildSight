<template>
  <main-layout>
  </main-layout>
  <div class="container">
    <img v-bind:src="imageSrc" class="background-img"/>
    <div class="centered-text">WildSight</div>
    <div class="subtitle">Conservation accelerated with AI</div>
  </div>
</template>

<script>
import MainLayout from '../layouts/Main.vue'
export default {
  components: {
    MainLayout
  },
  data() {
    return {
      imageSrc: "https://user-images.githubusercontent.com/31543169/111704559-565fbf80-880d-11eb-99d1-118a40bfd014.jpg"
    }
    
  }
}
</script>

<style lang="scss">
body, html {
  padding: 0;
  margin: 0;
  width: 100%;
  min-height: 100vh;
}
body {
  background-color: rgb(63, 63, 63);
}
.container {
  position: relative;
  text-align: center;
  color: rgb(63, 63, 63);
}
.background-img {
  width: 100%;
}
.centered-text {
  position: absolute;
  top: 10%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 60px;
  color: green;
}
.subtitle {
  position: absolute;
  top: 20%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 20px;
  color: green;
}
</style>