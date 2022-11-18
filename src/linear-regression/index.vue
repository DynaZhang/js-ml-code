<template>
  <div className="form-wrapper">
    <a-form :label-col="{ span: 8 }" :wrapper-col="{ span: 16 }">
      <a-form-item label="x">
        <a-input v-model:value="formData.x"></a-input>
      </a-form-item>
      <a-form-item :wrapper-col="{ offset: 8, span: 16 }">
        <a-button type="primary" :disabled="training" @click="onConfirm">{{
          btnText
        }}</a-button>
      </a-form-item>
      <a-form-item label="y">
        <a-input readonly v-model:value="resText" />
      </a-form-item>
    </a-form>
  </div>
</template>

<script setup lang="ts">
import * as tfvis from "@tensorflow/tfjs-vis";
import * as tfjs from "@tensorflow/tfjs";
import { computed, onMounted, reactive, ref, toRaw } from "vue";

let model: tfjs.Sequential;
const formData = reactive({
  x: "",
});
const training = ref<boolean>(true);
const resText = ref<number | undefined>(undefined);
const btnText = computed(() => {
  return training.value ? "训练中" : "预测";
});

const onConfirm = () => {
  const rowFormData = toRaw(formData);
  const outputs =
    model && model.predict(tfjs.tensor([parseFloat(rowFormData.x)]));
  resText.value = outputs.dataSync()[0];
};

onMounted(async () => {
  tfvis.visor().close();
  const xs = [1, 2, 3, 4];
  const ys = [1, 3, 5, 7];

    tfvis.render.scatterplot(
      {
        name: "线性回归训练集",
      },
      {
        values: xs.map((x, index) => ({ x, y: ys[index] })),
      },
      {
        xAxisDomain: [0, 5],
        yAxisDomain: [-1, 17],
      }
    );

  // 定义一个连续的模型（某一层的输入是其上一层的输出）
  model = tfjs.sequential();

  // Creates a dense (fully connected) layer.
  // This layer implements the operation: output = activation(dot(input, kernel) + bias)
  // 创建一个全连接层
  // 继承操作：dot（点乘）input(输入) kernel(权重) bias(偏置)
  model.add(
    tfjs.layers.dense({
      units: 1, // 定义一个神经元，输出空间的维度
      inputShape: [1], // 对应tfjs.tensor([...])中的shape属性
    })
  );

  // 给模型设置编译参数
  model.compile({
    loss: tfjs.losses.meanSquaredError, // 设置损失函数 均方误差
    optimizer: tfjs.train.sgd(0.1), // 设置优化器，指定学习速率（超参数） 随机梯度下降
  });

  //设置输入数据和标签
  const inputs = tfjs.tensor(xs);
  const labels = tfjs.tensor(ys);

  /**
   * batchSize和epochs都是超参数（超参数是编程人员在机器学习算法中用于调整的旋钮）
   * callbacks 可视化展示训练过程
   */
  await model.fit(inputs, labels, {
    batchSize: 4, // 学习的小批次样本数量
    epochs: 100, // 迭代训练数组的个数
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
