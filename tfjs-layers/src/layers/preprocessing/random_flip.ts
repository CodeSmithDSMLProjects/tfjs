/**
 * @license
 * Copyright 2023 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

 import { image, Rank, serialization, Tensor, tidy } from '@tensorflow/tfjs-core';
 import { getExactlyOneTensor, getExactlyOneShape } from '../../utils/types_utils';
 import {Shape} from '../../keras_format/common';
 import { Kwargs } from '../../types';
 import { ValueError } from '../../errors';
 import { BaseRandomLayerArgs, BaseRandomLayer } from '../../engine/base_random_layer';
 