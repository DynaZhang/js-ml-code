import { createApp } from "vue";
import * as VueRouter from "vue-router";
import "./style.css";
import App from "./App.vue";
import Antd from "ant-design-vue";
import "ant-design-vue/dist/antd.css";

import HomePage from "./home/index.vue";
import LinearRegression from "./linear-regression/index.vue";
import HeightWeight from "./height-weight/index.vue";
import LogisticsRegression from "./logistics-regression/index.vue";

const router = VueRouter.createRouter({
  history: VueRouter.createWebHistory(),
  routes: [
    {
      path: "/linear",
      component: LinearRegression,
    },
    {
      path: "/hw",
      component: HeightWeight,
    },
    {
      path: "/logistics",
      component: LogisticsRegression,
    },
    {
      path: "/home",
      component: HomePage,
    },
  ],
});

createApp(App).use(Antd).use(router).mount("#app");
