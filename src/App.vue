<template>
    <div class="top-bar">
        <router-link tag="a" to="/linear">线性回归</router-link>
        <router-link tag="a" to="/hw">归一化</router-link>
        <router-link tag="a" to="/logistics">逻辑回归</router-link>
        <router-link tag="a" to="/xor">xor逻辑回归</router-link>
        <router-link tag="a" to="/iris">鸢尾花问题</router-link>
        <router-link tag="a" to="/overfit">过拟合和欠拟合</router-link>
        <router-link tag="a" to="/cnn">识别手写数字</router-link>
        <router-link tag="a" to="/mobile-net">加载预训练模型</router-link>
        <router-link tag="a" to="/icon">图标识别</router-link>
    </div>
    <div class="content-wrapper">
        <router-view />
    </div>
</template>

<script setup lang="ts"></script>
<style scoped>
.top-bar {
    padding: 0 50px;
}
</style>
