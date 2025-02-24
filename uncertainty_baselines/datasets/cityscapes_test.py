# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Cityscapes."""

from absl.testing import parameterized
import tensorflow as tf
import uncertainty_baselines as ub


# TODO(dusenberrymw): Use TFDS mocking.
class CityscapesDatasetTest(ub.datasets.DatasetTest, parameterized.TestCase):

  # TODO(dusenberrymw): Rename to `test_dataset_size`.
  def testCityscapesDatasetShape(self):
    super()._testDatasetSize(
        ub.datasets.CityscapesDataset,
        image_size=(1024, 2048, 3),
        label_size=(1024, 2048, 1),
        validation_percent=0.1)

  def test_expected_features(self):
    builder = ub.datasets.ImageNetDataset('train')
    dataset = builder.load(batch_size=1)
    self.assertEqual(list(dataset.element_spec.keys()), ['features', 'labels'])

    builder_with_file_name = ub.datasets.ImageNetDataset(
        'train', include_file_name=True)
    dataset_with_file_name = builder_with_file_name.load(batch_size=1)
    self.assertEqual(
        list(dataset_with_file_name.element_spec.keys()),
        ['features', 'labels', 'file_name'])

if __name__ == '__main__':
  tf.test.main()
