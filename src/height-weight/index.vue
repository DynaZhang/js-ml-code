<template>
  <div className="form-wrapper">
    <a-form :label-col="{ span: 8 }" :wrapper-col="{ span: 16 }">
      <a-form-item label="身高">
        <a-input v-model:value="formData.height" suffix="cm"></a-input>
      </a-form-item>
      <a-form-item :wrapper-col="{ offset: 8, span: 16 }">
        <a-button type="primary" :disabled="training" @click="onConfirm">{{
          btnText
        }}</a-button>
      </a-form-item>
      <a-form-item label="预测体重">
        <a-input readonly v-model:value="resText" suffix="kg" />
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
  height: undefined,
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
      tfjs
        .tensor([parseFloat(rowFormData.height)])
        .sub(150)
        .div(20)
    );
  resText.value = outputs.mul(20).add(40).dataSync()[0];
};

onMounted(async () => {
  tfvis.visor().close();
  const heights = [150, 160, 170];
  const weights = [40, 50, 60];

  //   tfvis.render.scatterplot(
  //     { name: "身高体重训练数据" },
  //     {
  //       values: heights.map((height: number, index: number) => ({
  //         x: height,
  //         y: weights[index],
  //       })),
  //     },
  //     {
  //       xAxisDomain: [140, 220],
  //       yAxisDomain: [30, 100],
  //     }
  //   );

  const inputs = tfjs
    .tensor(heights)
    .sub(150)
    .div(170 - 150);
  const labels = tfjs
    .tensor(weights)
    .sub(40)
    .div(60 - 40);

  model = tfjs.sequential();

  model.add(
    tfjs.layers.dense({
      units: 1,
      inputShape: [1],
    })
  );

  model.compile({
    loss: tfjs.losses.meanSquaredError,
    optimizer: tfjs.train.sgd(0.1),
  });

  await model.fit(inputs, labels, {
    batchSize: 3,
    epochs: 100,
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
