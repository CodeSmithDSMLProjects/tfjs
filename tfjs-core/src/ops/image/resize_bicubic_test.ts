/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

import * as tf from '../../index';
import {ALL_ENVS, describeWithFlags} from '../../jasmine_util';
// import {expectArraysClose} from '../../test_util';

describeWithFlags('resizeBicubic', ALL_ENVS, () => {
  it('simple alignCorners=false', async () => {
    const input = tf.tensor4d([2,2,2,
                               1,1,1,
                               3,3,3], [1, 3, 3, 1]);


                                [0,0,0,0,0,0,0,
                                 0,0,0,0,0,0,0,
                                 0,0,2,2,2,0,0,
                                 0,0,1,1,1,0,0,
                                 0,0,3,3,3,0,0,
                                 0,0,0,0,0,0,0,
                                 0,0,0,0,0,0,0]

    console.log(`bicubic2 ${tf.image.resizeBicubic(input, [6, 9], false)}`);
    tf.tensor4d


  });

});
