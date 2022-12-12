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
  console.log(`${images.shape} images.shape`)
  const imagesStrides = util.computeStrides(images.shape);
  console.log(`imagesStrides ${imagesStrides}`)
  const [newHeight, newWidth] = size;
  const [batch, oldHeight, oldWidth, numChannels] = images.shape;

  oldHeight
  oldWidth
  const xValues = backend.data.get(images.dataId).values as TypedArray;


  function indexToCoordinates(point: number, imagesStrides: number[]): number[] {
    const rowPoint = Math.floor(point / imagesStrides[1])
    const colPoint = point % imagesStrides[1]
    return [rowPoint, colPoint]
  }

  function coordinatesToIndex(row: number, col:number, imagesStrides: number[]) {
    return (row * imagesStrides[1]) + col
  }


  function padInputImage(img: TypedArray): any {
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
      else if (col > imgCol + pad -1) {// case: right
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



// const topBotIdx = [...Array(imagesStrides[1]).keys()]
// const leftRightIdx = [...Array((imagesStrides[0] / imagesStrides[1])).keys()]

// let val: number
// let c1: number
// let c2: number
// let c3: number

// topBotIdx.forEach(el => {
//   val = img[coordinatesToIndex([0, el], imagesStrides)]
//   c1 = coordinatesToIndex([0, el + 2], paddedStrides)
//   c2 = coordinatesToIndex([1, el + 2], paddedStrides)
//   c3 = coordinatesToIndex([2, el + 2], paddedStrides)
//   console.log(` top: ${el+2}: ${[c1, c2, c3]} val ${val}`)
//   result[c1] = val
//   result[c2] = val
//   result[c3] = val
// })

// topBotIdx.forEach(el => {
// val = img[coordinatesToIndex([(imagesStrides[0] / imagesStrides[1]) - 1, el], imagesStrides)]
// c1 = coordinatesToIndex([(paddedStrides[0] / paddedStrides[1]) - 3, el + 2], paddedStrides)
// c2 = coordinatesToIndex([(paddedStrides[0] / paddedStrides[1]) - 2, el + 2], paddedStrides)
// c3 = coordinatesToIndex([(paddedStrides[0] / paddedStrides[1]) - 1, el + 2], paddedStrides)
// console.log(` bot: ${el+2}: ${[c1, c2, c3]} val ${val}`)
// result[c1] = val
// result[c2] = val
// result[c3] = val
// })

// leftRightIdx.forEach(el => {
// val = img[coordinatesToIndex([el, 0], imagesStrides)]
// c1 = coordinatesToIndex([el+2, 0], paddedStrides)
// c2 = coordinatesToIndex([el+2, 1], paddedStrides)
// c3 = coordinatesToIndex([el+2, 2], paddedStrides)
// console.log(` left: ${el+2}: ${[c1, c2, c3]} val ${val}`)
// result[c1] = val
// result[c2] = val
// result[c3] = val
// })

// leftRightIdx.forEach(el => {
// val = img[coordinatesToIndex([el, imagesStrides[1] - 1], imagesStrides)]
// c1 = coordinatesToIndex([el + 2, (paddedStrides[0] / paddedStrides[1]) - 2], paddedStrides)
// c2 = coordinatesToIndex([el + 2, (paddedStrides[0] / paddedStrides[1]) - 1], paddedStrides)
// c3 = coordinatesToIndex([el + 2, (paddedStrides[0] / paddedStrides[1]) ], paddedStrides)
// console.log(` right: ${el+2}: ${[c1, c2, c3]} val ${val}`)
// result[c1] = val
// result[c2] = val
// result[c3] = val

// })


// function insertValues(coord: number[], pad:number, img:TypedArray, loc:'t'|'b'|'l'|'r', paddedStrides: number[]) {
//   const idx = coordinatesToIndex(coord, imagesStrides)
//   const val = img[idx]
//   const padValues = [...Array(pad).keys()]

//   let resultIdx: number

//   if(loc == 't') {
//     padValues.forEach(el =>{
//       resultIdx = coordinatesToIndex([coord[0] + el, coord[1]], paddedStrides)
//       result[resultIdx] = val
//     })
//   }
//   if(loc == 'b') {
//     padValues.forEach(el =>{
//       resultIdx = coordinatesToIndex([coord[0] + (el + 1), coord[1]], paddedStrides)
//       result[resultIdx] = val
//     })
//   }
//   if(loc == 'l') {
//     padValues.forEach(el =>{
//       resultIdx = coordinatesToIndex([coord[0], coord[1] + el], paddedStrides)
//       result[resultIdx] = val
//     })
//   }
//   if(loc == 'r') {
//     padValues.forEach(el =>{
//       resultIdx = coordinatesToIndex([coord[0], coord[1] + (el+1)], paddedStrides)
//       result[resultIdx] = val
//     })
//   }


// topCoords.forEach(el => {
//   insertValues(el, pad, img, 't', paddedStrides)
// })

// botCoords.forEach(el => {
//   insertValues(el, pad, img, 'b', paddedStrides)
// })

// leftCoords.forEach(el => {
//   insertValues(el, pad, img, 'l', paddedStrides)
// })

// rightCoords.forEach(el => {
//   insertValues(el, pad, img, 'r', paddedStrides)
// })


/*
  const paddedMat = new Float32Array((height + pad)*(width + pad));
  imgLen = len(originalImg)
  const maxLen = len(paddedMat);
  for (let i = 0; i < maxLen; i++) {
    row, col = coordinatesToIndex(i, paddedStrides);
    if (row < pad) {
      // case: top
      if (col < pad) {
        // case: top left
        paddedMat[i] = originialImg[idxFromLoc(0, 0, inputStrides)]; // top left from original img
      }
      else if (col > imgLen + pad - 1) {
        // case: top right
        paddedMat[i] = originalImg[idxFromLoc(0, imgLen - 1, inputStrides)]; // top right from original img
      }
      else {
        // case: top, not corner
        paddedMat[i] = originalImg[idxFromLoc(0, col - pad, inputStrides)]; // top from original img
      }
    }
    else (row > imgLen + pad - 1) {
      // case: bottom
      if (col < pad) {
        // case: bottom left
        paddedMat[i] = originialImg[idxFromLoc(imgLen - 1, 0, inputStrides)]; // bottom left from original img
      }
      else if (col > imgLen + pad - 1) {
        // case: bottom right
        paddedMat[i] = originalImg[idxFromLoc(imgLen - 1, imgLen - 1, inputStrides)]; // bottom right from original img
      }
      else {
        // case: bottom, not corner
        paddedMat[i] = originalImg[idxFromLoc(imgLen - 1, col - pad, inputStrides)]; // bottom of original image
      }
    }
    else if (col < pad) {
      // left side
      paddedMat[i] = originalImg[idxFromLoc(row - pad, 0, inputStrides)]; // left of original image
    }
    else if (col > imgLen + pad -1) {
      // right side
      paddedMat[i] = originalImg[idxFromLoc(row - pad, imgLen - 1, inputStrides)]; // right of original image
    }
  }
*/



//
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




    // const pad = 2
    // const imageRows = imagesStrides[0] / imagesStrides[1]
    // const result = new Float32Array(
    //   util.sizeFromShape([batch, oldHeight + pad*2, oldWidth + pad*2, numChannels]));
    // const paddedStrides = util.computeStrides([batch, oldHeight + (pad*2), oldWidth + pad*2, numChannels])

    // let idx: number[]
    // let idx2: number
    // let val2: number

    // // pad with zeros
    // for(let i = 0; i < xValues.length; i++) {
    //   val2 = xValues[i]
    //   idx = indexToCoordinates(i, imagesStrides)
    //   idx = [idx[0] + pad, idx[1] + pad]
    //   idx2 = coordinatesToIndex(idx, paddedStrides)
    //   result[idx2] = val2
    // }

    // // top/bottom rows
    // const topBotIdx = [...Array(imagesStrides[1]).keys()]
    // const topCoords: number[][] = []
    // const botCoords:number[][] = []
    // topBotIdx.forEach(el => {
    //   topCoords.push([0, el])
    //   botCoords.push([imageRows - 1, el])
    // })

    // // left/right cols
    // const leftRightIdx = [...Array((imagesStrides[0] / imagesStrides[1])).keys()]
    // const leftCoords: number[][] = []
    // const rightCoords: number[][] = []
    // leftRightIdx.forEach(el => {
    //   leftCoords.push([el, 0])
    //   rightCoords.push([el, imagesStrides[1] - 1, ])
    // })

    // // need to get value of image at coordinate
    // // convert it to index
    // // if top row:
    //     // keep column same, subtract 1 and 2 from row
    // topCoords.forEach(el => {
    //   let idx = coordinatesToIndex(el, imagesStrides)
    //   let val = img[idx]
    //   let c1 = [0, el[1]+2]
    //   let c2 = [1, el[1]+2]
    //   console.log('t')
    //   console.log(c1)
    //   console.log(c2)
    //   let c1Idx = coordinatesToIndex(c1, paddedStrides)
    //   let c2Idx = coordinatesToIndex(c2, paddedStrides)
    //   console.log(c1Idx)
    //   console.log(c2Idx)
    //   console.log(`val ${val}`)
    //   result[c1Idx] = val
    //   result[c2Idx] = val
    // })


    // // if bottom row
    //     // keep column same, add 1 and 2 to row
    // botCoords.forEach(el => {
    //   let idx = coordinatesToIndex(el, imagesStrides)
    //   let val = img[idx]
    //   let c1 = [imageRows+pad, el[1]+2]
    //   let c2 = [imageRows+pad+1, el[1]+2]
    //   console.log('b')
    //   console.log(c1)
    //   console.log(c2)
    //   let c1Idx = coordinatesToIndex(c1, paddedStrides)
    //   let c2Idx = coordinatesToIndex(c2, paddedStrides)
    //   console.log(c1Idx)
    //   console.log(c2Idx)
    //   console.log(`val ${val}`)
    //   result[c1Idx] = val
    //   result[c2Idx] = val
    // })
    // // if left col
    // leftCoords.forEach(el => {
    //   let idx = coordinatesToIndex(el, imagesStrides)
    //   let val = img[idx]
    //   let c1 = [el[0]+2, 0]
    //   let c2 = [el[0]+2, 1]
    //   console.log('l')
    //   console.log(c1)
    //   console.log(c2)
    //   let c1Idx = coordinatesToIndex(c1, paddedStrides)
    //   let c2Idx = coordinatesToIndex(c2, paddedStrides)
    //   console.log(c1Idx)
    //   console.log(c2Idx)
    //   console.log(`val ${val}`)
    //   result[c1Idx] = val
    //   result[c2Idx] = val
    // })

    //   //  keep row same subtract 1 and 2 from column

    // // if right col
    // rightCoords.forEach(el => {
    //   let idx = coordinatesToIndex(el, imagesStrides)
    //   let val = img[idx]
    //   let c1 = [el[0]+2, imagesStrides[1] + 2]
    //   let c2 = [el[0]+2, imagesStrides[1] + 3]
    //   console.log('r')
    //   console.log(c1)
    //   console.log(c2)
    //   let c1Idx = coordinatesToIndex(c1, paddedStrides)
    //   let c2Idx = coordinatesToIndex(c2, paddedStrides)
    //   console.log(c1Idx)
    //   console.log(c2Idx)
    //   console.log(`val ${val}`)
    //   result[c1Idx] = val
    //   result[c2Idx] = val
    // })

