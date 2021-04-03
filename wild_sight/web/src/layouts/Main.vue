<template>
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container-fluid">
      <a class="navbar-brand" href="/">WildSight</a>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarSupportedContent">
        <ul class="navbar-nav">
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