<template>
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
  <div class="container-fluid">
    <button class="navbar-toggler" type="button"
    data-bs-toggle="collapse" data-bs-target="#navbar"
    aria-controls="navbarNav" aria-expanded="false"
    aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbar">
      <ul class="navbar-nav">
        <li class="nav-item">
          <v-link class="nav-link" aria-current="page" href="/">Home</v-link>
        </li>
        <li class="nav-item">
          <v-link class="nav-link" aria-current="page" href="/about">About</v-link>
        </li>
        <li class="nav-item">
          <v-link class="nav-link" aria-current="page" href="/demo">Demo</v-link>
        </li>
      </ul>
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