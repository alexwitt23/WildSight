<template>
  <nav class="navbar navbar-dark bg-dark flex-wrap">
  <div class="container-fluid">
    <button class="navbar-toggler flex-grow-sm-1 flex-grow-0 me-2"
      type="button" data-bs-toggle="collapse" data-bs-target="#navbar1">
      <span class="navbar-toggler-icon"></span>
    </button>
    <a class="navbar-brand flex-grow-1" to="/">WildSight</a>
    <div class="collapse navbar-collapse flex-grow-1 justify-content-center" id="navbar1">
      <ul class="navbar-nav mx-auto">
        <li class="nav-item">
          <router-link class="nav-link" to="/">Home</router-link>
        </li>
        <li class="nav-item">
          <router-link class="nav-link" to="/about">About</router-link>
        </li>
        <li class="nav-item">
          <router-link class="nav-link" to="/demo">Demo</router-link>
        </li>
      </ul>
    </div>
  </div>
</nav>
</template>

<script>
import 'bootstrap'
import 'bootstrap/dist/css/bootstrap.min.css'

export default {}
</script>