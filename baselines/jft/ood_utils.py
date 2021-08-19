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

"""Utils for OOD evaluation."""

import numpy as np


def compute_mean_and_cov(codes, labels):
  """Compute mean for each Guassian distribution, cov for all distributions."""
  dim = codes.shape[1]
  n_class = int(np.max(labels)) + 1
  mean_list = []
  cov = np.zeros((dim, dim))

  for class_id in range(n_class):
    codes_one_class = codes[labels == class_id]
    codes_mean_one_class = np.mean(codes_one_class, axis=0)
    cov += np.dot((codes_one_class - codes_mean_one_class).T,
                  (codes_one_class - codes_mean_one_class))
    mean_list.append(codes_mean_one_class)
  cov = cov / len(labels)
  return mean_list, cov


def compute_maha_dist(xs, mean_list, cov, epsilon=1e-20):
  """Mahalanobis distance between each of the input x and each of the class."""
  v = cov + np.eye(cov.shape[0], dtype=int) * epsilon  # avoid singularity
  vi = np.linalg.inv(v)

  n = xs.shape[0]
  n_class = len(mean_list)
  out = np.zeros((n, n_class))
  for i in range(n):
    x = xs[i]
    for class_id in range(n_class):
      mu = mean_list[class_id]
      out[i, class_id] = np.dot(np.dot((x - mu), vi), (x - mu).T)
  return out
