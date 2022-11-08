<template>
    <div class="top-bar">
        <router-link tag="a" to="/linear">线性回归</router-link>
        <router-link tag="a" to="/hw">归一化</router-link>
        <router-link tag="a" to="/logistics">逻辑回归</router-link>
        <router-link tag="a" to="/xor">xor</router-link>
        <router-link tag="a" to="/iris">iris</router-link>
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
