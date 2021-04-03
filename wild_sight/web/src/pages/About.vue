<template>
  <main-layout>
  </main-layout>
  <h2>About Us</h2>
  <p>
    This project was created as part of a senior design class at UT Austin.
    The goal is to empower biological conservationists with the power to accelerate
    their studies with AI. Our application delivers a deep neural network capable of
    detecting various classes of animals.
  </p>
  
</template>

<script>
import MainLayout from '../layouts/Main.vue'
export default {
  components: {
    MainLayout
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