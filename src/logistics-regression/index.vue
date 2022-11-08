<template>
  <div className="form-wrapper">
    <a-form :label-col="{ span: 8 }" :wrapper-col="{ span: 16 }">
      <a-form-item label="x">
        <a-input v-model:value="formData.x"></a-input>
      </a-form-item>
      <a-form-item label="y">
        <a-input v-model:value="formData.y"></a-input>
      </a-form-item>
      <a-form-item label="" :wrapper-col="{ offset: 8, span: 16 }">
        <a-button type="primary" :disabled="training" @click="onConfirm">{{
          btnText
        }}</a-button>
      </a-form-item>
      <a-form-item label="label">
        <a-input readonly v-model:value="resText"></a-input>
      </a-form-item>
    </a-form>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, toRaw } from "vue";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as tfjs from "@tensorflow/tfjs";
import { getData, Point } from "./data";

let model: tfjs.Sequential;
const formData = reactive({
  x: "",
  y: "",
});
const training = ref<boolean>(true);
const resText = ref<number | undefined>(undefined);
const btnText = computed(() => {
  return training.value ? "训练中" : "预测";
});

const onConfirm = () => {
  const rowFormData = toRaw(formData);
  const outputs =
    model &&
    model.predict(
      tfjs.tensor([[parseFloat(rowFormData.x), parseFloat(rowFormData.y)]])
    );
  resText.value = outputs.dataSync()[0];
};

onMounted(async () => {
  const data = getData(400);
  tfvis.render.scatterplot(
    { name: "逻辑回归训练数据" },
    {
      values: [
        data.filter((point: Point<number>) => point.label === 1),
        data.filter((point: Point<number>) => point.label === 0),
      ],
    }
  );

  model = tfjs.sequential();
  model.add(
    tfjs.layers.dense({
      units: 1,
      inputShape: [2],
      activation: "sigmoid", // 设置激活函数激活函数（Activation Function）是一种添加到人工神经网络中的函数，旨在帮助网络学习数据中的复杂模式。 类似于人类大脑中基于神经元的模型，激活函数最终决定了要发射给下一个神经元的内容。 在人工神经网络中，一个节点的激活函数定义了该节点在给定的输入或输入集合下的输出。），sigmoid函数保证输出的值在[0,1]之间
    })
  );

  model.compile({
    loss: tfjs.losses.logLoss, // 逻辑回归使用对数损失函数定义损失
    optimizer: tfjs.train.adam(), // adam与sgd都是随机梯度下降优化器，但adam可以不指定学习速率
  });

  const inputs = tfjs.tensor(
    data.map((point: Point<number>) => [point.x, point.y])
  );
  const labels = tfjs.tensor(data.map((point: Point<number>) => point.label));
  await model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 50,
    callbacks: tfvis.show.fitCallbacks({ name: "训练过程" }, ["loss"]),
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
