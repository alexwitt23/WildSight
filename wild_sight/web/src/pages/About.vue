<template>
  <main-layout>
  </main-layout>
    <div class="cover2">
        <div class="container">
            <h1 class="display-1 mt-5 text-center">About</h1>
        </div>
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