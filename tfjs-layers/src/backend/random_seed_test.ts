/**
 * @license
 * Copyright 2023 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {RandomSeed} from './random_seed';
import {describeMathCPUAndGPU} from '../utils/test_utils';

const randomSeed = new RandomSeed(42);

describeMathCPUAndGPU('RandomSeed', () => {
  it('Checking if RandomSeed class handles pseudo randomness.', () => {
    const firstSeed = randomSeed.currentSeed;
    randomSeed.next();
    const secondSeed = randomSeed.currentSeed;
    expect(firstSeed).not.toEqual(secondSeed);
  });
});
