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


  // function indexToCoordinates(point: number): number[] {
  //   const rowPoint = Math.floor(point / imagesStrides[1])
  //   const colPoint = point % imagesStrides[1]
  //   return [rowPoint, colPoint]
  // }

  function coordinatesToIndex(coord: number[], imagesStrides: number[]) {
    return (coord[0] * imagesStrides[1]) + coord[1]
  }


  function padInputImage(img: TypedArray) {

    const result = new Float32Array(
      util.sizeFromShape([batch, oldHeight + 4, oldWidth + 4, numChannels]));
    const paddedStrides = util.computeStrides([batch, oldHeight + 4, oldWidth + 4, numChannels])
    // const topLeftC = coordinatesToIndex([0,0])
    // const topRightC= coordinatesToIndex([0,imagesStrides[1] - 1])
    // const bottomLeftC = coordinatesToIndex([(imagesStrides[0] / imagesStrides[1]) - 1, 0])
    // const bottomRightC = coordinatesToIndex([(imagesStrides[0] / imagesStrides[1]) - 1, imagesStrides[1] - 1])

    const topBotIdx = Array.from({length: imagesStrides[1] - 2}, (_, i) => i + 1)
    const leftRightIdx = Array.from({length: (imagesStrides[0] / imagesStrides[1]) - 2}, (_, i) => i + 1)

    let val: number
    let c1: number
    let c2: number
    let c3: number

    topBotIdx.forEach(el => {
       val = img[coordinatesToIndex([0, el], imagesStrides)]
       c1 = coordinatesToIndex([0, el], paddedStrides)
       c2 = coordinatesToIndex([1, el], paddedStrides)
       c3 = coordinatesToIndex([2, el], paddedStrides)
       console.log(` top: ${el}: ${[c1, c2, c3]}`)
       result[c1] = val
       result[c2] = val
       result[c3] = val
    })

    topBotIdx.forEach(el => {
       val = img[coordinatesToIndex([(imagesStrides[0] / imagesStrides[1]) - 1, el], imagesStrides)]
       c1 = coordinatesToIndex([(paddedStrides[0] / paddedStrides[1]) - 1, el], paddedStrides)
       c2 = coordinatesToIndex([(paddedStrides[0] / paddedStrides[1]), el], paddedStrides)
       c3 = coordinatesToIndex([(paddedStrides[0] / paddedStrides[1]) + 1, el], paddedStrides)
       console.log(`${[c1, c2, c3]}`)

       result[c1] = val
       result[c2] = val
       result[c3] = val
    })

    leftRightIdx.forEach(el => {
      val = img[coordinatesToIndex([el, 0], imagesStrides)]
      c1 = coordinatesToIndex([el, 0], paddedStrides)
      c2 = coordinatesToIndex([el, 1], paddedStrides)
      c3 = coordinatesToIndex([el, 2], paddedStrides)
      console.log(`${[c1, c2, c3]}`)

      result[c1] = val
      result[c2] = val
      result[c3] = val
    })

    leftRightIdx.forEach(el => {
      val = img[coordinatesToIndex([el, imagesStrides[1] - 1], imagesStrides)]
      c1  = coordinatesToIndex([el, paddedStrides[1] - 1], paddedStrides)
      c2  = coordinatesToIndex([el, paddedStrides[1]], paddedStrides)
      c3  = coordinatesToIndex([el, paddedStrides[1] + 1], paddedStrides)
      console.log(`${[c1, c2, c3]}`)
      result[c1] = val
      result[c2] = val
      result[c3] = val
    })

    return result

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



    // // const result = new Float32Array(
    // //   util.sizeFromShape([batch, oldHeight + 4, oldWidth + 4, numChannels]));
    // // first/last rows and first/last cols
    // //first row [[0,1] [0,2] --- [0, imagestrides[1] - 1]]
    // let firstrow: number[][]
    // firstrow =[]
    // const frCoord = Array.from({length: imagesStrides[1] - 2}, (_, index) => index + 1);
    // console.log(`firstrow`)
    // frCoord.forEach(el => {
    //   console.log([0, el])
    //   firstrow.push([0, el])
    // })

    // const lastrowpoint = Math.ceil(img.length / imagesStrides[1]) - 1
    // let lastrow: number[][]
    // lastrow =[]
    // const lrCoord = Array.from({length: imagesStrides[1] - 2}, (_, index) => index + 1);
    // console.log(`lastrow`)
    // lrCoord.forEach(el => {

    //   console.log([lastrowpoint, el])
    //   firstrow.push([lastrowpoint, el])
    // })

    // //last row [[lastrowpoint,1] [lastrowpoint,2] --- [lastrowpoint, imagestrides[1] - 1]]
    // //  firstcol [[1,0] [2,0] -- [lastrowpoint, 0]]
    // let firstcol: number[][]
    // firstcol = []
    // const fcCoord = Array.from({length: lastrowpoint}, (_, index) => index + 1);
    // console.log(`firstcol`)
    // fcCoord.forEach(el => {
    //   console.log([el, 0])
    //   firstcol.push([el, 0])
    // })

    // //lastcol [[1,imagestrides[1] - 1] [2,imagestrides[1] - 1] -- [lastrowpoint, imagestrides[1] - 1]]
    // let lastcol: number[][]
    // lastcol = []
    // const lcCoord = Array.from({length: lastrowpoint}, (_, index) => index + 1);
    // console.log(`lastcol`)
    // lcCoord.forEach(el => {
    //   console.log([el, imagesStrides[1] - 1])
    //   firstcol.push([el, imagesStrides[1] - 1])
    // })

    // console.log(`firstrow ${firstrow}`)
    // console.log(`lastrow ${lastrow}`)
    // console.log(`firstcol ${firstcol}`)
    // console.log(`lastcol ${lastcol}`)

    // let idx;
    // let val: number;
    // let paddedCoords
    // let paddedidx

    // firstrow.forEach(el => {
    //   idx = coordinatesToIndex(el)
    //   console.log(`firstrow`)
    //   console.log(`idx ${idx}`)
    //   val = img[idx]
    //   console.log(`val ${val}`)
    //   paddedCoords = [[0, el[1]], [1, el[1]]]
    //   paddedCoords.forEach(el => {
    //     paddedidx = coordinatesToIndex(el)
    //     result[paddedidx] = val
    //   })
    // })

    // lastrow.forEach(el => {
    //   console.log(`lastrow`)
    //   idx = coordinatesToIndex(el)
    //   console.log(`idx ${idx}`)
    //   val = img[idx]
    //   console.log(`val ${val}`)
    //   paddedCoords = [[lastrowpoint + 1, el[1]], [lastrowpoint + 2, el[1]]]
    //   paddedCoords.forEach(el => {
    //     paddedidx = coordinatesToIndex(el)
    //     result[paddedidx] = val
    //   })
    // })

    // firstcol.forEach(el => {
    //   console.log(`firstcol`)
    //   idx = coordinatesToIndex(el)
    //   console.log(`idx ${idx}`)
    //   val = img[idx]
    //   console.log(`val ${val}`)
    //   paddedCoords = [[el[0], 0], [el[0], 1]]
    //   paddedCoords.forEach(el => {
    //     paddedidx = coordinatesToIndex(el)
    //     result[paddedidx] = val
    //   })
    // })

    // lastcol.forEach(el => {
    //   console.log(`lastcol`)
    //   idx = coordinatesToIndex(el)
    //   console.log(`idx ${idx}`)
    //   val = img[idx]
    //   console.log(`val ${val}`)
    //   paddedCoords = [[el[0], imagesStrides[1]], [el[0], imagesStrides[1] + 1]]
    //   paddedCoords.forEach(el => {
    //     paddedidx = coordinatesToIndex(el)
    //     result[paddedidx] = val
    //   })
    // })

