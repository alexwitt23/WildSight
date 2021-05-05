import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
    { path: '/', component: () => import('./pages/Home') },
    { path: '/about', component: () => import('./pages/About') },
    { path: '/demo', component: () => import('./pages/Demo') },
    { path: '/swift-parrot', component: () => import('./pages/Parrots') },
]

const router = createRouter({
    history: createWebHashHistory(),
    routes,
  })
  
export default router