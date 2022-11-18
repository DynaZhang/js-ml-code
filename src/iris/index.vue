<template>
<div className="form-wrapper">
    <a-form :model="formData" :label-col="{span: 8}" :wrapper-col="{span: 16}">
        <a-form-item label="花萼长度">
            <a-input v-model:value="formData.eHeight" suffix="cm" />
        </a-form-item>
        <a-form-item label="花萼宽度">
            <a-input v-model:value="formData.eWidth" suffix="cm" />
        </a-form-item>
        <a-form-item label="花瓣长度">
            <a-input v-model:value="formData.bHeight" suffix="cm" />
        </a-form-item>
        <a-form-item label="花瓣宽度">
            <a-input v-model:value="formData.bWidth" suffix="cm" />
        </a-form-item>
        <a-form-item label="" :wrapper-col="{offset: 8, span: 16}">
            <a-button type="primary" :disabled="training" @click="onConfirm">{{btnText}}</a-button>
        </a-form-item>
        <a-form-item label="鸢尾花种类">
            <p>{{resText}}</p>
        </a-form-item>
    </a-form>
</div>
</template>

<script setup lang="ts">
import {computed, onMounted, reactive, ref, toRaw} from 'vue';
import * as tfjs from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {getIrisData, IRIS_CLASSES} from './data';

let model: tfjs.Sequential;

const formData = reactive({
    eHeight: '',
    eWidth: '',
    bHeight: '',
    bWidth: ''
});
const training = ref<boolean>(true);
const btnText = computed(() => {
    return training.value ? '训练中' : '预测';
});
const resText = ref<string>('');

const onConfirm = () => {
    const {eHeight, eWidth, bHeight, bWidth} = toRaw(formData);
    const input = tfjs.tensor([[parseFloat(eHeight), parseFloat(eWidth), parseFloat(bHeight), parseFloat(bWidth)]]);
    const output = model.predict(input);
    resText.value = IRIS_CLASSES[output.argMax(1).dataSync()[0]] // argMax取list中的最大值，根据yTrain的数据格式[[x,y,z]],需要取第二维的list（索引从0开始）
}

onMounted(async () => {
    tfvis.visor().close();
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

    training.value = false;
});
</script>

<style scoped>
.form-wrapper {
    margin: 50px;
    width: 500px;
}
</style>
