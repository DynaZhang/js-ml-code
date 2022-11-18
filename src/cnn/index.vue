<template>
    <div class="container">
        <div>
            <a-button type="primary" @click="onTrainModel">训练模型</a-button>
            <a-button type="primary" style="margin-left: 8px" @click="onLoadModel">加载预训练模型</a-button>
            <a-button type="primary" style="margin-left: 8px" @click="onDownloadModel" :disabled="training || !loadedModel">下载模型</a-button>
            
        </div>
        <template v-if="!(training || !loadedModel)">
            <canvas
                class="test-canvas"
                ref="canvasRef"
                width="300"
                height="300"
                @mousedown="onMouseDown"
                @mouseup="onMouseUp"
                @mouseleave="onMouseLeave"
                @mousemove="onMouseMove"
            />
            <div>预测结果：{{ resText || '-' }}</div>
            <div>
            <a-button type="primary" style="margin-right: 8px" @click="onPredictResult" :disabled="training || !loadedModel">预测结果</a-button>
            <a-button type="primary" @click="onClearCanvas">清除</a-button>
        </div>
        </template>
    </div>
</template>

<script setup lang="ts">
import * as tfjs from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { message } from 'ant-design-vue';
import CryptoJS from 'crypto-js';
import {nextTick, ref} from 'vue';
import {MnistData} from './data';

const parsedWordArray = CryptoJS.enc.Base64.parse('aHR0cDovL2R5bmF0ZXN0LmJqLmJjZWJvcy5jb20vanMtbWwvY25uL2Nubl9udW1iZXIuanNvbg==');
const MODEL_PATH = parsedWordArray.toString(CryptoJS.enc.Utf8);
let model: tfjs.Sequential | tfjs.LayersModel | undefined;

const resText = ref<string>('');
const training = ref<boolean>(false);
const loadedModel = ref<boolean>(false);

const canvasRef = ref<any>(null);
let canvasContext: CanvasRenderingContext2D | null = null;
const isMouseClick = ref<boolean>(false);

const onClearCanvas = () => {
    canvasContext?.clearRect(0, 0, 300, 300);
    canvasContext.fillStyle = '#000';
    canvasContext?.fillRect(0, 0, 300, 300);
};

const onMouseDown = () => {
    isMouseClick.value = true;
};
const onMouseUp = () => {
    isMouseClick.value = false;
};
const onMouseMove = (e: MouseEvent) => {
    if (!isMouseClick.value) {
        return;
    }
    canvasContext?.beginPath();
    canvasContext?.arc(e.offsetX, e.offsetY, 10, 0, 2 * Math.PI);
    canvasContext?.closePath();
    canvasContext.fillStyle = '#fff';
    canvasContext?.fill();
};
const onMouseLeave = () => {
    isMouseClick.value = false;
};

const onTrainModel = async () => {
    tfvis.visor().close();
    model = undefined;
    training.value = true;
    const data = new MnistData();
    await data.load();
    const examples = data.nextTestBatch(20);
    const surface = tfvis.visor().surface({name: '输入数据示例'}); // 定义一个surface来显示训练数据
    const container = document.getElementById('container');
    for (let i = 0; i < 20; i++) {
        const imageTensor = tfjs.tidy(() => {
            return examples.xs.slice([i, 0], [1, 784]).reshape([28, 28]); // 提取一个黑白图片的像素值，每张图片大小是28 * 28
        });
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style.margin = '4px';
        await tfjs.browser.toPixels(imageTensor, canvas);
        // container && container.appendChild(canvas);
        surface.drawArea.appendChild(canvas);
    }

    model = tfjs.sequential();
    model.add(
        tfjs.layers.conv2d({
            inputShape: [28, 28, 1],
            kernelSize: 5,
            filters: 8, // 超参数
            strides: 1, // 设置移动步长
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        })
    );
    model.add(
        tfjs.layers.maxPooling2d({
            poolSize: [2, 2],
            strides: [2, 2]
        })
    );
    model.add(
        tfjs.layers.conv2d({
            kernelSize: 5,
            filters: 16,
            strides: 1,
            activation: 'relu', // max(x, 0);
            kernelInitializer: 'varianceScaling'
        })
    );
    model.add(
        tfjs.layers.maxPooling2d({
            poolSize: [2, 2],
            strides: [2, 2]
        })
    );
    model.add(tfjs.layers.flatten()); // 把高维数据摊平，然后放到dense层做处理
    model.add(
        tfjs.layers.dense({
            units: 10, //输出0-9
            activation: 'softmax',
            kernelInitializer: 'varianceScaling'
        })
    );

    model.compile({
        loss: 'categoricalCrossentropy', // 交叉熵损失函数
        optimizer: tfjs.train.adam(),
        metrics: 'accuracy'
    });

    // 使用tidy将多个tensor操作合并
    const [trainXS, trainYS] = tfjs.tidy(() => {
        const d = data.nextTrainBatch(1000);
        console.log(d);
        return [d.xs.reshape([1000, 28, 28, 1]), d.labels]; //d.xs的shape是[1000, 784]，需要进行reshape操作与cnn的输入数据保持一致
    });
    const [textXS, testYS] = tfjs.tidy(() => {
        const d = data.nextTestBatch(200);
        return [d.xs.reshape([200, 28, 28, 1]), d.labels]; //d.xs的shape是[200, 784]，需要进行reshape操作与cnn的输入数据保持一致
    });
    await model.fit(trainXS, trainYS, {
        epochs: 50, // 超参数
        validationData: [textXS, testYS],
        callbacks: tfvis.show.fitCallbacks({name: '训练效果'}, ['loss', 'val_loss', 'acc', 'val_acc'], {
            callbacks: ['onEpochEnd']
        })
    });
    training.value = false;
    loadedModel.value = true;
    nextTick(() => {
        canvasContext = canvasRef.value.getContext('2d') as CanvasRenderingContext2D;
        onClearCanvas();
    });
};

const onDownloadModel = async () => {
    await model?.save('downloads://cnn_number');
    message.success('下载成功');
};

const onLoadModel = async () => {
    model = await tfjs.loadLayersModel(MODEL_PATH);
    message.success('加载成功');
    loadedModel.value = true;
    nextTick(() => {
        canvasContext = canvasRef.value.getContext('2d') as CanvasRenderingContext2D;
        onClearCanvas();
    });
}

const onPredictResult = () => {
    const input = tfjs.tidy(() => {
        return tfjs.image
            .resizeBilinear(tfjs.browser.fromPixels(canvasRef.value), [28, 28], true)
            .slice([0, 0, 0], [28, 28, 1])
            .toFloat()
            .div(255)
            .reshape([1, 28, 28, 1]);
    });

    const pred = model?.predict(input).argMax(1);
    resText.value = pred.dataSync()[0];
};
</script>

<style scoped>
.container {
    margin: 50px;
}
.test-canvas {
    margin-top: 20px;
}
</style>
