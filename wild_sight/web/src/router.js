import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
    { path: '/', component: () => import('./pages/Home') },
    { path: '/about', component: () => import('./pages/About') },
    { path: '/demo', component: () => import('./pages/Demo') },
]

const router = createRouter({
    history: createWebHashHistory(),
    routes,
  })
  
export default router