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
  private adjustedHeight: number;
  private adjustedWidth: number;

  constructor(args: RandomWidthArgs) {
    super(args);
    this.height = args.height;
    this.width = args.width;
    this.seed = args.seed;
  }
}
