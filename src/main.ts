import {createApp} from 'vue';
import * as VueRouter from 'vue-router';
import './style.css';
import App from './App.vue';
import Antd from 'ant-design-vue';
import 'ant-design-vue/dist/antd.css';

import LinearRegression from './linear-regression/index.vue';
import HeightWeight from './height-weight/index.vue';
import LogisticsRegression from './logistics-regression/index.vue';
import XORPage from './xor/index.vue';
import IrisPage from './iris/index.vue';

const router = VueRouter.createRouter({
    history: VueRouter.createWebHistory(),
    routes: [
        {
            path: '/linear',
            component: LinearRegression
        },
        {
            path: '/hw',
            component: HeightWeight
        },
        {
            path: '/logistics',
            component: LogisticsRegression
        },
        {
            path: '/xor',
            component: XORPage
        },
        {
            path: '/iris',
            component: IrisPage
        }
    ]
});

createApp(App).use(Antd).use(router).mount('#app');
