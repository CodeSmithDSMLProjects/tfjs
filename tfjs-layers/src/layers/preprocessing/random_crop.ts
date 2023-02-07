/**
 * @license
 * Copyright 2023 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import { BaseRandomLayerArgs, BaseRandomLayer } from '../../engine/base_random_layer';
import { image, serialization, Tensor, tidy } from '@tensorflow/tfjs-core';
import {Shape} from '../../keras_format/common';
import { getExactlyOneTensor, getExactlyOneShape } from '../../utils/types_utils';

export declare interface RandomWidthArgs extends BaseRandomLayerArgs {
  height: number;
  width: number;
  seed?: number; // default = false;
}

// A preprocessing layer which randomly crops images during training.
export class RandomCrop extends BaseRandomLayer {
  /** @nocollapse */
  static override className = 'RandomCrop';
  private readonly height: number;
  private readonly width: number;
  private readonly seed?: number; // default null

  constructor(args: RandomWidthArgs) {
    super(args);
    this.height = args.height;
    this.width = args.width;
    this.seed = args.seed;
  }

  override getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      'height': this.height,
      'width': this.width,
      'seed': this.seed
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
