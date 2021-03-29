<template>
  <ul>
    <li><v-link href="/">Home</v-link></li>
    <li><v-link href="/about">About</v-link></li>
    <li><v-link href="/demo">Demo</v-link></li>
  </ul>

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