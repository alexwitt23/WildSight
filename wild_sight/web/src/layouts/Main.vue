<template>
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
  <div class="container-fluid">
    <div class="collapse navbar-collapse" id="navbarNav">
      <ul class="navbar-nav">
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
import VLink from '../components/VLink.vue'
export default {
  components: {
    VLink
  }
}
</script>