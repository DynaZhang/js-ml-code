<template></template>

<script setup lang="ts">
import {onMounted} from 'vue';
import * as tfjs from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {getIrisData, IRIS_CLASSES} from './data';

let model: tfjs.Sequential;
onMounted(async () => {
    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.2);
    model = tfjs.sequential();

    model.add(
        tfjs.layers.dense({
            units: 10,
            inputShape: [xTrain.shape[1] as number],
            activation: 'sigmoid'
        })
    );

    model.add(
        tfjs.layers.dense({
            units: 3,
            activation: 'softmax'
        })
    );

    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tfjs.train.adam(0.1),
        metrics: ['accuracy']
    });

    await model.fit(xTrain, yTrain, {
        epochs: 100,
        validationData: [xTest, yTest],
        callbacks: tfvis.show.fitCallbacks({name: 'iris数据训练过程'}, ['loss', 'val_loss', 'acc', 'val_acc'], {
            callbacks: ['onEpochEnd']
        })
    });
});
</script>

<style scoped></style>
