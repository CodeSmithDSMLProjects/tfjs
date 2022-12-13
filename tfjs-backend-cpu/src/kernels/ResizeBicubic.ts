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
  console.log(`halfPixelCenters ${halfPixelCenters}`)

  assertNotComplex(images, 'resizeBicubic');
  const imagesStrides = util.computeStrides(images.shape);
  const [newHeight, newWidth] = size;
  const [batch, oldHeight, oldWidth, numChannels] = images.shape;

  const xValues = backend.data.get(images.dataId).values as TypedArray;
  console.log(`xValues ${xValues}`)
  console.log(`imagesStrides ${imagesStrides}`)

  function indexToCoordinates(point: number, imagesStrides: number[]): number[] {
    const rowPoint = Math.floor(point / imagesStrides[1])
    const colPoint = point % imagesStrides[1]
    return [rowPoint, colPoint]
  }

  function coordinatesToIndex(row: number, col:number, imagesStrides: number[]) {
    return (row * imagesStrides[1]) + col
  }

  function padInputImage(img: TypedArray): Float32Array {
    const pad = 2;
    const paddedMat = new Float32Array(util.sizeFromShape([batch, oldHeight + pad*2, oldWidth + pad*2, numChannels]));
    const paddedStrides = util.computeStrides([batch, oldHeight + pad*2, oldWidth + pad*2, numChannels]);
    const [imgRow, imgCol] = [(imagesStrides[0] / imagesStrides[1]), imagesStrides[1]];
    const maxLen = paddedMat.length;
    for (let i = 0; i < maxLen; i++) {
      const [row, col] = indexToCoordinates(i, paddedStrides);
      const topRow = 0;
      const bottomRow = imgRow - 1;
      const leftCol = 0;
      const rightCol = imgCol - 1;
      const currentCol = col - pad;
      const currentRow = row - pad;
      if (row < pad) { // case: top
        if (col < pad) { // case: top left
          paddedMat[i] = img[coordinatesToIndex(topRow, leftCol, imagesStrides)]; // top left from original img
        }
        else if (col > imgCol + pad - 1) { // case: top right
          paddedMat[i] = img[coordinatesToIndex(topRow, rightCol, imagesStrides)]; // top right from original img
        }
        else { // case: top, not corner
          paddedMat[i] = img[coordinatesToIndex(topRow, currentCol, imagesStrides)]; // top from original img
        }
      }
      else if (row > imgRow + pad - 1) { // case: bottom
        if (col < pad) { // case: bottom left
          paddedMat[i] = img[coordinatesToIndex(bottomRow, leftCol, imagesStrides)]; // bottom left from original img
        }
        else if (col > imgCol + pad - 1) { // case: bottom right
          paddedMat[i] = img[coordinatesToIndex(bottomRow, rightCol, imagesStrides)]; // bottom right from original img
        }
        else { // case: bottom, not corner
          paddedMat[i] = img[coordinatesToIndex(bottomRow, currentCol, imagesStrides)]; // bottom of original image
        }
      }
      else if (col < pad) { // case: left
        paddedMat[i] = img[coordinatesToIndex(currentRow, leftCol, imagesStrides)]; // left of original image
      }
      else if (col > imgCol + pad -1) { // case: right
        paddedMat[i] = img[coordinatesToIndex(currentRow, rightCol, imagesStrides)]; // right of original image
      }
      else { // case: center
        paddedMat[i] = img[coordinatesToIndex(currentRow, currentCol, imagesStrides)]; // original image value
      }
    }
    return paddedMat;
  }


  const result = padInputImage(xValues)
  console.log(`result ${result}`)

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


  return backend.makeTensorInfo(
    [batch, newHeight, newWidth, numChannels], 'float32', result);

}

export const resizeBicubicConfig: KernelConfig = {
  kernelName: ResizeBicubic,
  backendName: 'cpu',
  kernelFunc: resizeBicubic as unknown as KernelFunc
};

