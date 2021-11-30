# Diabetic Retinopathy

In this baseline, models try to predict the presence or absence of diabetic
retinopathy (a binary classification task) using data from the
[Kaggle Diabetic Retinopathy Detection challenge](https://www.kaggle.com/c/diabetic-retinopathy-detection/data). Please see
that page for details on data collection, etc.

Models are trained with images of blood vessels in the eye, as seen in the
[TensorFlow Datasets description](https://www.tensorflow.org/datasets/catalog/diabetic_retinopathy_detection).

## Model Checkpoints
For each method we release the best-performaing checkpoints. These checkpoints were trained on the combined training and validation set, using hyperparameters selected from the best validation performance. Each checkpoint was selected to be from the step during training with the best test AUC (averaged across the 10 random seeds). This was epoch 63 for the deterministic model, epoch 72 for the MC-Dropout method, epoch 31 for the Variational Inference method, and epoch 61 for the Radial BNNs method. For more details on the models, see the accompanying [Model Card](./model_card.md), which covers all the models below, as the dataset is exactly the same across them all, and the only model differences are minor calibration improvements. The checkpoints can be browsed [here](https://console.cloud.google.com/storage/browser/gresearch/reliable-deep-learning/checkpoints/baselines/diabetic_retinopathy_detection).

## Tuning
For this baseline, two rounds of quasirandom search were conducted on the hyperparameters listed below, where the first round was a heuristically-picked larger search space and the second round was a hand-tuned smaller range around the better performing values. Each round was for 50 trials, and the final hyperparemeters were selected using the final validation AUC from the second tuning round. These best hyperparameters were used to retrain combined train and validation sets over 10 seeds. **We note that the learning rate schedules could likely be tuned for improved performance, but leave this to future work.** All our intermediate and final tuning results are available below hosted on [tensorboard.dev](tensorboard.dev).

Below are links to [tensorboard.dev](tensorboard.dev) TensorBoards for each baseline method that contain the metric values of the various tuning runs as well as the hyperparameter points sampled in the `HPARAMS` tab at the top of the page.

#### Deterministic
[[First Tuning Round]](https://tensorboard.dev/experiment/nAygVvdjSWWAEQRDD8Z0Aw/) [[Final Tuning Round]](https://tensorboard.dev/experiment/GLxGQR8pQhypBr9jGdBMUQ/)  [[Best Hyperparamters 10 seeds]](https://tensorboard.dev/experiment/lh5yXcwzRc2ZNmId34ujPw/)

---

#### Monte-Carlo Dropout
[[First Tuning Round]](https://tensorboard.dev/experiment/xDVLkDAgR1uJqyxIqkdPIQ/) [[Final Tuning Round]](https://tensorboard.dev/experiment/1qy7JJfYQYqQ1lanieSYew/)  [[Best Hyperparamters 10 seeds]](https://tensorboard.dev/experiment/aMr4glcES6qg43P4HvckTg/)

---

#### Radial Bayesian Neural Networks
[[First Tuning Round]](https://tensorboard.dev/experiment/5CzJYikVTvKQLdqSnmUrpg/) [[Final Tuning Round]](https://tensorboard.dev/experiment/RDf1PKZkSZ2PGo1H8wnWBw/)  [[Best Hyperparamters 10 seeds]](https://tensorboard.dev/experiment/040rBdKBQPir8cDhReyk3A/)

---

#### Variational Inference
[[First Tuning Round]](https://tensorboard.dev/experiment/gVwRJIRoQoyRrfG1boJVPA/) [[Final Tuning Round]](https://tensorboard.dev/experiment/n9NYA7ryRG6jCYdpyQYoOQ/)  [[Best Hyperparamters 10 seeds]](https://tensorboard.dev/experiment/mPZt9k0lQ1yF2TAuE2cxqw/)

---


### Search spaces
Search space for the initial and final rounds of tuning on the deterministic method. We used a stepwise decay for the initial round but switched to a linear decay for the final round to alleviate overfitting, where we tuned the linear decay factor on the grid `[1e-3, 1e-2, 0.1]`.

| | Learning Rate | 1 - momentum | L2 |
|---|---|---|---|
| Initial | [1e-3,0.1] | [1e-2,0.1] | [1e-5,1e-3] |
| Final | [0.03, 0.5] | [5e-3, 0.05] | [1e-6, 2e-4] |

Search space for the initial and final rounds of tuning on the Monte Carlo Dropout method.

| | Learning Rate | 1 - momentum | L2 | dropout |
|---|---|---|---|---|
| Initial | [1e-3,0.1] | [1e-2,0.1] | [1e-5,1e-3] | [0.01, 0.25] |
| Final | [1e-2,0.5] | [1e-2, 0.04] | [1e-5, 1e-3] | [0.01, 0.2]  |

Search space for the initial and final rounds of tuning on the Radial BNN method.

| | Learning Rate | 1 - momentum | L2 | stddev_mean_init | stddev_stddev_init |
|---|---|---|---|---|---|
| Initial | [1e-3,0.1] | [1e-2,0.1] | [1e-5,1e-3] | [1e-5,1e-1] | [1e-2,1] |
| Final | [0.15,1] | [1e-2, 0.05] | [1e-4, 1e-3] | [1e-5, 2e-2] | [1e-2, 0.2] |

Search space for the initial and final rounds of tuning on the Variational Inference method.

| | Learning Rate | 1 - momentum | L2 | stddev_mean_init | stddev_stddev_init |
|---|---|---|---|---|---|
| Initial | [1e-3,0.1] | [1e-2,0.1] | [1e-5,1e-3] | [1e-5,1e-1] | [1e-2,1] |
| Final | [0.02,5] | [0.02, 0.1] | [1e-5, 2e-4] | [1e-5, 2e-3] | [1e-2, 1] |

## Cite

Please cite our paper if you use this code in your own work:

```
@article{filos2019systematic,
  title={A Systematic Comparison of Bayesian Deep Learning Robustness in Diabetic Retinopathy Tasks},
  author={Filos, Angelos and Farquhar, Sebastian and Gomez, Aidan N and Rudner, Tim GJ and Kenton, Zachary and Smith, Lewis and Alizadeh, Milad and de Kroon, Arnoud and Gal, Yarin},
  journal={arXiv preprint arXiv:1912.10481},
  year={2019}
}
```

## Deferred Prediction

### Task Description
In Deferred Prediction, a model's predictive uncertainty is used to choose a subset of the test set for which predictions will be evaluated. In particular, the uncertainty per test input forms a ranking, and the model's performance is evaluated on the X% of test inputs with the least uncertainty. X is referred to as the _retain percentage_, and the other (100 - X)% of the data is _deferred_. Standard evaluation therefore uses a _retain fraction_ = [1], i.e., the full test set is retained.

### Real-World Relevance
We may wish to use a predictive model of diabetic retinopathy to ease the burden on clinical practitioners. Under deferred prediction, the model refers the examples on which it is least confident to expert doctors. We can tune the _retain fraction_ parameter based on practitioner availability, and a model with well-calibrated uncertainty will have high performance on metrics such as AUC/accuracy on the retained evaluation data, because its uncertainty and predictive performance are correlated.

### Usage
* Given: trained models with varied random seeds
1.  For each model, compute deferred prediction metric results with `run_deferred_prediction.py`.
    * The user should specify a path to the model checkpoint (`--checkpoint_dir`), its training random seed (`--train_seed`), and the type of model (e.g., 'dropout', `--model_type`). The script can be run for different models/seeds with a consistent output directory (e.g., `--output_dir='gs://uncertainty-baselines/deferred_prediction_results`). The user can set an array of retain fractions, e.g., [0.5, 0.6, 0.7, 0.8, 0.9, 1], with the `--deferred_prediction_fractions` hyperparameter.
    * Example usage:
        ```
      python baselines/diabetic_retinopathy_detection/run_deferred_prediction.py --data_dir='/path/to/retinopathy/data' --num_cores=1 --use_gpu=True --checkpoint_dir='gs://uncertainty-baselines/variational_inference_checkpoint' --train_seed=42 --model_type=variational_inference --output_dir='gs://uncertainty-baselines/deferred_prediction_results'
        ```
2. Plot results with `plot_deferred_prediction.py`.
    * Each plot will contain deferred prediction curves for a particular metric, and one or many model types.
    * To generate a plot for a particular model type:
        ```
        python baselines/diabetic_retinopathy_detection/plot_deferred_prediction.py --results_dir='gs://uncertainty-baselines/deferred_prediction_results/variational_inference' --plot_dir='.' --model_type=variational_inference
        ```
      where the `--results_dir` should point to the subdirectory of a particular type of model, as generated by `run_deferred_prediction.py`.
    * To generate a plot for all model types stored under the `--results_dir` (and also found in the `DEFERRED_PREDICTION_MODEL_TYPES` list in the` plot_deferred_prediction.py` file):
        ```
        python baselines/diabetic_retinopathy_detection/plot_deferred_prediction.py --results_dir='gs://uncertainty-baselines/deferred_prediction_results' --plot_dir='.'
        ```

## Acknowledgements

The Diabetic Retinopathy Detection baseline was contributed through collaboration with the [Oxford Applied and Theoretical Machine Learning](http://oatml.cs.ox.ac.uk/) (OATML) group, with sponsorship from:

<table align="center">
    <tr>
        <td><img src="https://github.com/OATML/bdl-benchmarks/blob/alpha/assets/intel.png" style="float: left; width: 200px; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em"></td>
        <td><img src="https://github.com/OATML/bdl-benchmarks/blob/alpha/assets/oatml.png" style="float: left; width: 200px; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em"></td>
        <td><img src="https://github.com/OATML/bdl-benchmarks/blob/alpha/assets/oxcs.png" style="float: left; width: 200px; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em"></td>
        <td><img src="https://github.com/OATML/bdl-benchmarks/blob/alpha/assets/turing.png" style="float: left; width: 200px; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em"></td>
    </tr>
</table>
