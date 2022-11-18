<template>
    <div class="form-wrapper">
        <a-form :model="formData" :label-col="{span: 8}" :wrapper-col="{span: 16}">
            <a-form-item label="x">
                <a-input v-model:value="formData.x" />
            </a-form-item>
            <a-form-item label="y">
                <a-input v-model:value="formData.y" />
            </a-form-item>
            <a-form-item :wrapper-col="{offset: 8, span: 16}">
                <a-button type="primary" :disabled="training" @click="onConfirm">{{ btnText }}</a-button>
            </a-form-item>
            <a-form-item label="x">
                <a-input v-model:value="resText" />
            </a-form-item>
        </a-form>
    </div>
</template>

<script setup lang="ts">
import * as tfvis from '@tensorflow/tfjs-vis';
import * as tfjs from '@tensorflow/tfjs';
import {computed, onMounted, reactive} from 'vue';
import {PointLabel} from '../types/common';
import {getData} from './data';
import {ref, toRaw} from 'vue';

let model: tfjs.Sequential;
const formData = reactive({
    x: '',
    y: ''
});
const training = ref<boolean>(true);
const resText = ref<string>('');
const btnText = computed(() => {
    return training.value ? '训练中' : '预测';
});

const onConfirm = () => {
    const {x, y} = toRaw(formData);
    const inputs = tfjs.tensor([[parseFloat(x), parseFloat(y)]]);
    const output = model && model.predict(inputs);
    resText.value = output.dataSync()[0];
};

onMounted(async () => {
    tfvis.visor().close();
    const data = getData(400);
    tfvis.render.scatterplot(
        {name: 'xor训练数据集'},
        {
            values: [
                data.filter((point: PointLabel<0 | 1>) => point.label === 0),
                data.filter((point: PointLabel<0 | 1>) => point.label === 1)
            ]
        }
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
            activation: 'sigmoid'
        })
    );

    model.compile({
        loss: tfjs.losses.logLoss,
        optimizer: tfjs.train.adam(0.1)
    });

    const inputs = tfjs.tensor(data.map((point: PointLabel<0 | 1>) => [point.x, point.y]));
    const labels = tfjs.tensor(data.map((point: PointLabel<0 | 1>) => [point.label]));

    await model.fit(inputs, labels, {
        batchSize: 40,
        epochs: 10,
        callbacks: tfvis.show.fitCallbacks({name: 'xor训练过程'}, ['loss'])
    });
    training.value = false;
});
</script>

<style scoped>
.form-wrapper {
    width: 500px;
    margin: 50px;
}
</style>
