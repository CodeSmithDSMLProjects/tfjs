/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {KernelConfig, KernelFunc, ResizeBicubic, ResizeBicubicAttrs, ResizeBicubicInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function resizeBicubic(args: {
  inputs: ResizeBicubicInputs,
  backend: MathBackendCPU,
  attrs: ResizeBicubicAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {images} = inputs;
  const {alignCorners, halfPixelCenters, size} = attrs;
  console.log(`alignCorners ${alignCorners}`)
  console.log(`alignCorners ${halfPixelCenters}`)

  assertNotComplex(images, 'resizeBicubic');
  console.log(`${images.shape} images.shape`)
  const imagesStrides = util.computeStrides(images.shape);
  console.log(`imagesStrides ${imagesStrides}`)

  const [newHeight, newWidth] = size;

  const [batch, oldHeight, oldWidth, numChannels] = images.shape;
  console.log(`oldHeight ${oldHeight}`)
  console.log(`oldWidth ${oldWidth}`)

  const xValues = backend.data.get(images.dataId).values as TypedArray;

  const result = new Float32Array(
    util.sizeFromShape([batch, newHeight, newWidth, numChannels]));

  console.log(`result ${result}`)
  console.log(`xValues ${xValues}`)

  // const effectiveInputSize: [number, number] = [
  //   (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
  //   (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
  // ];

  // const effectiveOutputSize: [number, number] = [
  //   (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
  //   (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
  // ];

  // Interpolation kernel
  function interpolationKernel(x: number, a: number): number {
    if (Math.abs(x) >= 0 && Math.abs(x) <= 1) {
      return (a + 2) * (Math.abs(x) ** 3) - (a + 3) * (Math.abs(x) ** 2) + 1;
    }
    else if (Math.abs(x) > 1 && Math.abs(x) <= 2) {
      return a * (Math.abs(x)**3)-(5*a)*(Math.abs(x)**2)+(8*a)*Math.abs(x)-4*a;
    }
    return 0;
  }

  console.log(`interpolationKernel ${interpolationKernel(1,2)}`)

  function padInputImage(img:TypedArray, padSize: number[]): any {
    // initialize a zero-filled array of the correct final dimensions


    return img
  }


  const paddedTensor = padInputImage(xValues, [3,3]);
  console.log(`zeros ${paddedTensor}`)

  // Padding
  // def padding(img, H, W, C):
  //   zimg = np.zeros((H+4, W+4, C))
  //   zimg[2:H+2, 2:W+2, :C] = img
  //
  //   # Pad the first/last two col and row
  //   zimg[2:H+2, 0:2, :C] = img[:, 0:1, :C]
  //   zimg[H+2:H+4, 2:W+2, :] = img[H-1:H, :, :]
  //   zimg[2:H+2, W+2:W+4, :] = img[:, W-1:W, :]
  //   zimg[0:2, 2:W+2, :C] = img[0:1, :, :C]
  //
  //   # Pad the missing eight points
  //   zimg[0:2, 0:2, :C] = img[0, 0, :C]
  //   zimg[H+2:H+4, 0:2, :C] = img[H-1, 0, :C]
  //   zimg[H+2:H+4, W+2:W+4, :C] = img[H-1, W-1, :C]
  //   zimg[0:2, W+2:W+4, :C] = img[0, W-1, :C]
  //
  //   return zimg

  return backend.makeTensorInfo(
    [batch, newHeight, newWidth, numChannels], 'float32', result);

}

export const resizeBicubicConfig: KernelConfig = {
  kernelName: ResizeBicubic,
  backendName: 'cpu',
  kernelFunc: resizeBicubic as unknown as KernelFunc
};


// let output = new Float32Array(
    //   util.sizeFromShape(
    //       [batch, newHeight + padSize[0], newWidth + padSize[1], numChannels]
    //     )
    //   );

    //   for(let i = 0; i < img.length; i++) {
    //     if(i == 0) {


    //     }
    //   }
    // take edge values and expand them outwards to pad
    // for (let i = 0; i < oldHeight; i++) {
    //   for (let j = 0; j < oldWidth; j++) {


          // if j == 0 // top left corner
            // for x < pad
              // for y < pad
                // output[x][y] = input[i][j]
          // if j == oldWidth // top right corner
          // else // top edge, pad upwards
        // if j == 0
          // left edge, pad leftward
        // i == oldHeight
          // if j == 0 // bottom left corner
          // if j == oldWidth // bottom right corner
          // else // bottom edge, pad below
        // j == oldWidth
          // right edge, pad right

        // else: central values, output[i][j] = input[i][j]
          // output[i + pad][j + pad] = input[i][j]
      // }
    // }
