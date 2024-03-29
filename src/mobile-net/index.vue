<template>
    <div class="container">
        <div class="upload-wrapper">
            <a-upload
                list-type="picture-card"
                :before-upload="onBeforeUpload"
                @remove="onRemoveFile"
            >
                <a-button :disabled="loadedModel">
                    <upload-outlined></upload-outlined>
                    Select File
                </a-button>
            </a-upload>
        </div>
        <div>预测结果：{{ resText }}</div>
    </div>
</template>

<script setup lang="ts">
import {onMounted, ref} from 'vue';
import CryptoJS from 'crypto-js';
import * as tfjs from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import {IMAGENET_CLASSES} from './web_model/imagenet_classes';
import {file2Image} from '../utils/common';
import { message } from 'ant-design-vue';

const parsedWordArray = CryptoJS.enc.Base64.parse('aHR0cDovL2R5bmF0ZXN0LmJqLmJjZWJvcy5jb20vanMtbWwvbW9iaWxlLW5ldC9tb2RlbC5qc29u');
const MODEL_PATH = parsedWordArray.toString(CryptoJS.enc.Utf8);

let model: tfjs.LayersModel;

const resText = ref<string>('');
const loadedModel =  ref<boolean>(false);
const onBeforeUpload = async (file: File, fileList: any) => {
    onPredict(file);
    return false;
};

const onPredict = async (file: File) => {
    const image = await file2Image(file);
    const pred = tfjs.tidy(() => {
        // 转为浮点数，通过减除操作归一化成[-1,1]区间，然后reshape
        const input = tfjs.browser
            .fromPixels(image)
            .toFloat()
            .sub(255 / 2)
            .div(255 / 2)
            .reshape([1, 224, 224, 3]);
        return model.predict(input);
    });
    const index = pred.argMax(1).dataSync()[0] as number;
    resText.value = IMAGENET_CLASSES[index] as string || '';
};

const onRemoveFile = () => {};

onMounted(async () => {
    tfvis.visor().close();
    model = await tfjs.loadLayersModel(MODEL_PATH);
    message.success('加载成功');
});
</script>

<style scoped>
.container {
    margin: 50px;
}
</style>
