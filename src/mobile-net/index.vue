<template>
    <div class="container">
        <div class="upload-wrapper">
            <a-upload
                list-type="picture-card"
                :file-list="fileList"
                :before-upload="onBeforeUpload"
                @remove="onRemoveFile"
            >
                <a-button>
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
import * as tfjs from '@tensorflow/tfjs';
import {IMAGENET_CLASSES} from './web_model/imagenet_classes';

const MODEL_PATH = 'http://localhost:8080/model.json';
let model: tfjs.LayersModel;

const resText = ref('');

const file2Image = (file: File): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = (e: ProgressEvent<FileReader>) => {
            const image = new Image();
            image.src = e.target?.result as string;
            image.width = 224;
            image.height = 224;
            resolve(image);
        };
    });
};

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
    resText.value = IMAGENET_CLASSES[index];
};

const onRemoveFile = () => {};

onMounted(async () => {
    model = await tfjs.loadLayersModel(MODEL_PATH);
});
</script>

<style scoped>
.container {
    margin: 50px;
}
</style>
