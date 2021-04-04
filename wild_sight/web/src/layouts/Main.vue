<template>
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
  <div class="container-fluid">
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarTogglerDemo01" aria-controls="navbarTogglerDemo01" aria-expanded="false" aria-label="Toggle navigation">
      <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarTogglerDemo01">
      <ul class="navbar-nav me-auto mb-2 mb-lg-0">
        <li class="nav-item">
          <v-link class="nav-link active" aria-current="page" href="/">Home</v-link>
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