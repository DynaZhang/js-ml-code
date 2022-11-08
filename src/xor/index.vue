<template></template>

<script setup lang="ts">
import * as tfvis from '@tensorflow/tfjs-vis';
import * as tfjs from '@tensorflow/tfjs';
import { onMounted } from '@vue/runtime-core';
import { PointLabel } from '../types/common';
import {getData} from './data';

let model: tfjs.Sequential;
onMounted(async () => {
    const data = getData(400);
    tfvis.render.scatterplot(
        {name: 'xor训练数据集'},
        {values: [
            data.filter((point: PointLabel<0 | 1>) => point.label === 0),
            data.filter((point: PointLabel<0 | 1>) => point.label === 1)
        ]}
    );
    
    model = tfjs.sequential();
    model.add(
        tfjs.layers.dense({
            units: 4,
            inputShape: [2],
            activation: 'relu'
        })
    );
    // 不用设置inputSize，因为输入取决于上一个model.add的输出层
    model.add(
        tfjs.layers.dense({
            units: 1,
            activation: 'relu'
        })
    );

    model.compile({
        loss: tfjs.losses.logLoss,
        optimizer: tfjs.train.adam(0.1)
    });

    const inputs = tfjs.tensor(data.map((point: PointLabel<0 | 1>) => ([point.x, point.y])));
    const labels = tfjs.tensor(data.map((point: PointLabel<0 | 1>) => ([point.label])));

    await model.fit(inputs, labels, {
        batchSize: 40,
        epochs: 10,
        callbacks: tfvis.show.fitCallbacks({name: 'xor训练过程'}, ['loss'])
    });
});
</script>

<style scoped>

</style>