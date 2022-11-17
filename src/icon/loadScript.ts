import {loadImg} from '../utils/common';
import {BrandTrainDataSet} from '../types/common';

export const getInputs = async (): Promise<BrandTrainDataSet> => {
    const promiseList: Array<Promise<HTMLImageElement>> = [];
    const labels: Array<number[]> = [];
    for (let i = 0; i < 30; i++) {
        ['android', 'apple', 'windows'].forEach((label: string) => {
            const imgUrl = `http://dynatest.bj.bcebos.com/js-ml/brand/train/${label}-${i}.jpg`;
            const promise = loadImg(imgUrl);
            labels.push([label === 'android' ? 1 : 0, label === 'apple' ? 1 : 0, label === 'windows' ? 1 : 0]);
            promiseList.push(promise);
        });
    }
    const inputs = await Promise.all(promiseList);
    return {
        inputs,
        labels
    };
};
