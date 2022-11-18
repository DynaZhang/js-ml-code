<template></template>

<script setup lang="ts">
import {onMounted} from 'vue';
import * as tfjs from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import * as overfit from './data';
import * as xor from '../xor/data';
import {PointLabel} from '../types/common';

let model: tfjs.Sequential;

onMounted(() => {
    tfvis.visor().close();
    const overfitData = overfit.getData(200, 2);
    // const xorData = xor.getData(500);

    // tfvis.render.scatterplot(
    //     {name: 'xor训练数据'},
    //     {values: [
    //         overfitData.filter((point: PointLabel<0 | 1>) => point.label === 0),
    //         overfitData.filter((point: PointLabel<0 | 1>) => point.label === 1)
    //     ]}
    // );

    tfvis.render.scatterplot(
        {name: '带有干扰的训练数据'},
        {values: [
            overfitData.filter((point: PointLabel<0 | 1>) => point.label === 0),
            overfitData.filter((point: PointLabel<0 | 1>) => point.label === 1)
        ]}
    );

    model = tfjs.sequential();
    // model.add(tfjs.layers.dense({
    //     units: 1,
    //     inputShape: [2],
    //     activation: 'sigmoid'
    // }));
    model.add(tfjs.layers.dense({
        units: 10,
        inputShape: [2],
        activation: 'tanh',
        // kernelRegularizer: tfjs.regularizers.l2({l2: 1}) // 权重衰减（也是超参数）
    }));

    model.add(
        tfjs.layers.dropout({rate: 0.5})  // 假设有10个神经元权重，每次训练时随机丢弃10*0.5 = 5个神经元权重
    );

    model.add(tfjs.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
    model.compile({
        loss: tfjs.losses.logLoss,
        optimizer: tfjs.train.adam(0.1)
    });

    // const inputs = tfjs.tensor(xorData.map((point: PointLabel<0 | 1>) => [point.x, point.y]));
    // const labels = tfjs.tensor(xorData.map((point: PointLabel<0 | 1>) => point.label));

    const inputs = tfjs.tensor(overfitData.map((point: PointLabel<0 | 1>) => [point.x, point.y]));
    const labels = tfjs.tensor(overfitData.map((point: PointLabel<0 | 1>) => point.label));


    model.fit(inputs, labels, {
        validationSplit: 0.2,
        epochs: 200,  // 早停法，可以把epoch设置少一点
        callbacks: tfvis.show.fitCallbacks({name: '拟合结果'}, ['loss', 'val_loss'], {callbacks: ['onEpochEnd']})
    });
});
</script>

<style scoped>

</style>