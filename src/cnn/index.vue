<template>
    <div id="container"></div>
</template>

<script setup lang="ts">
import * as tfjs from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {onMounted} from 'vue';
import {MnistData} from './data';

onMounted(async () => {
    const data = new MnistData();
    await data.load();
    const examples = data.nextTestBatch(20);
    for (let i = 0; i < 20; i++) {
        const imageTensor = tfjs.tidy(() => {
            return examples.xs.slice([i, 0], [1, 784]).reshape([28, 28]);  // 提取一个黑白图片的像素值，每张图片大小是28 * 28
        });
        const surface = tfvis.visor().surface({name: '输入数据示例'});   // 定义一个surface来显示训练数据
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px';
        await tfjs.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas); // 将canvas放到surface
    }
});
</script>

<style scoped>
#container {
    margin: 50px;
}
</style>