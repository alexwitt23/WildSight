<template>
  <nav class="navbar navbar-dark bg-dark flex-wrap">
  <div class="container-fluid">
    <button class="navbar-toggler flex-grow-sm-1 flex-grow-0 me-2"
      type="button" data-bs-toggle="collapse" data-bs-target="#navbar1">
      <span class="navbar-toggler-icon"></span>
    </button>
    <a class="navbar-brand flex-grow-1" href="/">WildSight</a>
    <div class="collapse navbar-collapse flex-grow-1 justify-content-center" id="navbar1">
      <ul class="navbar-nav mx-auto">
        <li class="nav-item">
          <v-link class="nav-link" aria-current="page" href="/">Home</v-link>
        </li>
        <li class="nav-item">
          <v-link class="nav-link" href="/about">About</v-link>
        </li>
        <li class="nav-item">
          <v-link class="nav-link" href="/demo">Demo</v-link>
        </li>
      </ul>
    </div>
  </div>
</nav>
<slot></slot>
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