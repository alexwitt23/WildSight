<template>
  <nav class="navbar navbar-dark bg-dark flex-wrap">
    <div class="container-fluid">
      <button class="navbar-toggler flex-grow-sm-1 flex-grow-0 me-2"
      type="button" data-bs-toggle="collapse" data-bs-target="#navbar5">
        <span class="navbar-toggler-icon"></span>
      </button>

      <a class="navbar-brand flex-grow-1" href="/">WildSight</a>
     <div class="navbar-collapse collapse flex-grow-1 justify-content-center" id="navbar5">
        <ul class="navbar-nav mx-auto">
          <li class="nav-item nav-link">
            <v-link href="/">Home</v-link>
          </li>
          <li class="nav-item nav-link">
            <v-link href="/about">About</v-link>
          </li>
          <li class="nav-item nav-link">
            <v-link href="/demo">Demo</v-link>
          </li>
        </ul>
      </div>
        <div class="flex-grow-1">
            <!--spacer-->
        </div>
    </div>
  </nav>
  
</template>

<script>
import 'bootstrap'
import 'bootstrap/dist/css/bootstrap.min.css'
import VLink from '../components/VLink.vue'
export default {
  components: {
    VLink
  }
}
</script>

<style lang="scss">
@media (min-width: 768px) {
.navbar-brand.abs
    {
        position: absolute;
        width: 100%;
        left: 0;
        text-align: center;
    }
}
.navbar-toggler:focus,
.navbar-toggler:active,
.navbar-toggler-icon:focus {
    outline: none;
    box-shadow: none;
}
</style>