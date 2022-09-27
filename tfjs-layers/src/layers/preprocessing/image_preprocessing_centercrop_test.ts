import { tensor, Tensor3D, Tensor4D} from '@tensorflow/tfjs-core';
import { CenterCrop } from './image_preprocessing_centercrop';
import { describeMathCPUAndGPU, expectTensorsClose } from '../../utils/test_utils';

describeMathCPUAndGPU('CenterCrop Layer', () => {

  it("Crops batched image with even crop lengths as expected", () => {
    const rangeArr = [...Array(16).keys()]; // equivalent to np.arange(0,16)
    const inputArr = [];
    while(rangeArr.length) inputArr.push(rangeArr.splice(0,4)); // reshape
    const inputTensor = tensor([inputArr], [1,4,4,1]) as Tensor4D;
    const height = 2;
    const width = 2;
    const expectedOutput = tensor([[[5,6], [9,10]]], [1,2,2,1]) as Tensor4D;
    const centerCrop = new CenterCrop({height, width});
    const actualOutput = centerCrop.apply(inputTensor) as Tensor4D;
    expectTensorsClose(actualOutput, expectedOutput);
  })

  it("Crops batched image with odd crop lengths as expected", () => {
    const rangeArr = [...Array(16).keys()]; // equivalent to np.arange(0,16)
    const inputArr = [];
    while(rangeArr.length) inputArr.push(rangeArr.splice(0,4)); // reshape
    const inputTensor = tensor([inputArr], [1,4,4,1]) as Tensor4D;
    const height = 3;
    const width = 3;
    const expectedOutput = tensor([[[1,2,3], [5,6,7], [9,10,11]]], [1,3,3,1]) as Tensor4D;
    const centerCrop = new CenterCrop({height, width});
    const actualOutput = centerCrop.apply(inputTensor) as Tensor4D;
    expectTensorsClose(actualOutput, expectedOutput);
  })

  it('Upsizes image when crop boundaries are larger than image dims', () => {
    const inputArr =  [ [ [4 , 7],
                          [21, 9] ],

                        [ [8 , 9],
                          [1 ,33] ] ];
    const inputTensor = tensor([inputArr], [1,2,2,2]) as Tensor4D;
    const height = 3;
    const width = 4;
    const centerCrop = new CenterCrop({height, width});
    const layerOutputTensor = centerCrop.apply(inputTensor) as Tensor4D;
    const expectedArr = [[[4,7], [12.5, 8], [21, 9],[21, 9]],
    [[6.666667 , 8.333333 ], [7.1666665, 16.666666], [7.666666 , 25], [7.666666 , 25]],
   [[8, 9],[4.5, 21],[1, 33],[1, 33]]];
    const expectedOutput = tensor([expectedArr], [1,3,4,2]) as Tensor4D;
    expectTensorsClose(layerOutputTensor, expectedOutput);
  });

  it("Crops unbatched image with even crop lengths as expected", () => {
    const rangeArr = [...Array(16).keys()]; // equivalent to np.arange(0,16)
    const inputArr = [];
    while(rangeArr.length) inputArr.push(rangeArr.splice(0,4)); // reshape
    console.log(`inputArray ${inputArr}`)
    const inputTensor = tensor(inputArr, [4,4,1]) as Tensor3D;
    const height = 2;
    const width = 2;
    const expectedOutput = tensor([[5,6], [9,10]], [2,2,1]) as Tensor3D;
    const centerCrop = new CenterCrop({height, width});
    const actualOutput = centerCrop.apply(inputTensor) as Tensor3D;
    expectTensorsClose(actualOutput, expectedOutput);
  })

  it("Crops unbatched image with odd crop lengths as expected", () => {
    const rangeArr = [...Array(16).keys()]; // equivalent to np.arange(0,16)
    const inputArr = [];
    while(rangeArr.length) inputArr.push(rangeArr.splice(0,4)); // reshape
    const inputTensor = tensor(inputArr, [4,4,1]) as Tensor3D;
    const height = 3;
    const width = 3;
    const expectedOutput = tensor([[1,2,3], [5,6,7], [9,10,11]], [3,3,1]) as Tensor3D;
    const centerCrop = new CenterCrop({height, width});
    const actualOutput = centerCrop.apply(inputTensor) as Tensor3D;
    expectTensorsClose(actualOutput, expectedOutput);
  })
});
