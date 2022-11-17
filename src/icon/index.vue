<template>
    <div class="container">
        <div>
            <a-button type="primary" @click="onTrainModel">训练模型</a-button>
            <a-button type="primary" style="margin-left: 8px" @click="onDownloadModel"
                :disabled="training || !loadedModel">下载模型
            </a-button>
        </div>
        <div class="upload-wrapper">
            <a-upload list-type="picture-card" :before-upload="onBeforeUpload">
                <a-button :disabled="training || !loadedModel">
                    <upload-outlined></upload-outlined>
                    Select File
                </a-button>
            </a-upload>
        </div>
        <div>
            预测结果：{{ resText }}
            <a-button type="primary" style="margin-left: 8px" @click="onResetResult"
                :disabled="training || !loadedModel">
                重置结果
            </a-button>
        </div>
    </div>
</template>

<script setup lang="ts">
import * as tfvis from '@tensorflow/tfjs-vis';
import * as tfjs from '@tensorflow/tfjs';
import { message } from 'ant-design-vue';
import { ref } from 'vue';
import { getInputs } from './loadScript';
import { file2Image, img2x } from '../utils/common';

const MOBILE_NET_MODEL_PATH = 'http://dynatest.bj.bcebos.com/js-ml/mobile-net/model.json';
const MODEL_PATH = 'http://dynatest.bj.bcebos.com/js-ml/brand/model/brand.json';
const NUMBER_CLASSES = 3;

let model: tfjs.Sequential | tfjs.LayersModel;
let truncatdMobileNet: tfjs.LayersModel;

const resText = ref<string>('');
const loadedModel = ref<boolean>(false);
const training = ref<boolean>(false);

const onBeforeUpload = async (file: File, fileList: any) => {
    onPredict(file);
    return false;
};

const onTrainModel = async () => {
    training.value = true;
    const result = await getInputs();
    const surface = tfvis.visor().surface({
        name: '输入示例',
        styles: {
            height: '250px'
        }
    });
    result.inputs.forEach((imgElement: HTMLImageElement) => {
        surface.drawArea.appendChild(imgElement);
    });
    const mobileNetModel = await tfjs.loadLayersModel(MOBILE_NET_MODEL_PATH);
    // mobileNetModel.summary(); // 观察模型的结构
    const layer = mobileNetModel.getLayer('conv_pw_13_relu'); // 获取某个中间层
    // 模型截断
    truncatdMobileNet = tfjs.model({
        inputs: mobileNetModel.inputs,
        outputs: layer.output
    });

    model = tfjs.sequential();
    model.add(
        tfjs.layers.flatten({
            inputShape: layer.outputShape.slice(1) // [null, 7, 7, 256] null表示不确定
        })
    );
    model.add(
        tfjs.layers.dense({
            units: 10, // 超参数
            activation: 'relu'
        })
    );
    model.add(
        tfjs.layers.dense({
            units: NUMBER_CLASSES,
            activation: 'softmax'
        })
    );
    model.compile({
        loss: 'categoricalCrossentropy', // 交叉熵损失函数
        optimizer: tfjs.train.adam()
    });

    const { xs, ys } = tfjs.tidy(() => {
        const xs = tfjs.concat(
            result.inputs.map((imgElement: HTMLImageElement) => truncatdMobileNet.predict(img2x(imgElement)))
        );
        const ys = tfjs.tensor(result.labels);
        return { xs, ys };
    });

    await model.fit(xs, ys, {
        epochs: 20,
        callbacks: tfvis.show.fitCallbacks({ name: '训练效果' }, ['loss'], { callbacks: ['onEpochEnd'] })
    });

    training.value = false;
    loadedModel.value = true;
}

const onDownloadModel = async () => {
    await model?.save('downloads://icon');
    message.success('下载成功');
};

const onPredict = async (file: File) => {
    const image = await file2Image(file);
    const pred = tfjs.tidy(() => {
        // 转为浮点数，通过减除操作归一化成[-1,1]区间，然后reshape
        const x = img2x(image);
        const input = truncatdMobileNet.predict(x);
        return model.predict(input);
    });
    const index = pred.argMax(1).dataSync()[0] as number;
    resText.value = ['android', 'apple', 'windows'][index] as string || '';
};

const onResetResult = () => {
    resText.value = '';
}
</script>

<style scoped>
.container {
    margin: 50px;
}

.upload-wrapper {
    margin-top: 20px;
}
</style>

