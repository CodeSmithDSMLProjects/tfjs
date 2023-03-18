/**
 * @license
 * Copyright 2023 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {BaseRandomLayerArgs,BaseRandomLayer} from '../../engine/base_random_layer';
import {image,serialization,Tensor,Tensor3D,Tensor4D,tidy,DataType,stack} from '@tensorflow/tfjs-core';
import {Shape} from '../../keras_format/common';
import {Kwargs} from '../../types';
import {getExactlyOneShape, getExactlyOneTensor} from '../../utils/types_utils';
import * as K from '../../backend/tfjs_backend';

const {resizeBilinear, cropAndResize} = image;

export declare interface RandomCropArgs extends BaseRandomLayerArgs {
  height: number;
  width: number;
}

// A preprocessing layer which randomly crops images during training.
export class RandomCrop extends BaseRandomLayer {
  /** @nocollapse */
  static override className = 'RandomCrop';
  private readonly height: number;
  private readonly width: number;

  constructor(args: RandomCropArgs) {
    super(args);
    this.height = args.height;
    this.width = args.width;
  }

  randomCrop(inputs: Tensor3D | Tensor4D, hBuffer: number, wBuffer: number,
    height: number, width: number, inputHeight: number,
    inputWidth: number, dtype: DataType): Tensor | Tensor[] {

    return tidy(() => {
      return inputs; // temp return
    });
  }

  resize(inputs: Tensor3D | Tensor4D, height: number,
         width: number, dtype: DataType): Tensor | Tensor[] {

    return tidy(() => {
      const outputs = resizeBilinear(inputs, [height, width]);
      return K.cast(outputs, dtype);
    });
  }

  override call(inputs: Tensor3D | Tensor4D, kwargs: Kwargs):
      Tensor | Tensor[] {
    return tidy(() => {
      const rankedInputs = getExactlyOneTensor(inputs) as Tensor3D | Tensor4D;
      const dtype = rankedInputs.dtype;
      const inputShape = rankedInputs.shape;
      const inputHeight = inputShape[inputShape.length - 3];
      const inputWidth = inputShape[inputShape.length - 2];

      let hBuffer = 0;
      if (inputHeight !== this.height) {
        hBuffer = Math.floor((inputHeight - this.height) / 2);
      }

      let wBuffer = 0;
      if (inputWidth !== this.width) {
        wBuffer = Math.floor((inputWidth - this.width) / 2);

        if(wBuffer === 0) {
          wBuffer = 1;
        }
      }

      if(hBuffer >= 0 && wBuffer >= 0) {
        return this.randomCrop(rankedInputs, hBuffer, wBuffer,
                              this.height, this.width, inputHeight,
                              inputWidth, dtype);
      } else {
        return this.resize(inputs, this.height, this.width, dtype);
      }
    });
  }

  override getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      'height': this.height,
      'width': this.width,
    };

    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config
  }

  override computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = getExactlyOneShape(inputShape);
    const numChannels = inputShape[2];
    return [this.height, this.width, numChannels];
  }
}

serialization.registerClass(RandomCrop);
