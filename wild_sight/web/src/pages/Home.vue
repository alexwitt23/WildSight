<template>
  <navbar></navbar>
  <div class="cover">  
   <div class="container-fluid">
      <div class="row">
        <h1 class="text-center display-1 mt-5 title">WildSight</h1>
        <h3 class="text-center">Accelerating conservation with AI</h3>
      </div>
    </div>
  </div>

</template>

<script>
import Navbar from '../layouts/NavBar'
export default {
  components: {
    'navbar': Navbar
  }
}
</script>