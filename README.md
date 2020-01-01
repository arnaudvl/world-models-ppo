# world-models-ppo

World model implementation with PPO in PyTorch. This repository builds on [world-models](https://github.com/ctallec/world-models) for the model implementation and [firedup](https://github.com/kashif/firedup) for the PPO optimization.

First save a number of the *CarRacing-v0* Gym environment rollouts used for the train and test sets in the ```filepath``` folder:

```bash
python env/carracing.py --filepath './env/data' ---n_fold_train 10 ---n_fold_test 1
```

Then train the [Variational Autoencoder](https://arxiv.org/abs/1312.6114) using the stored rollout:

```bash
python vae/train.py --data_dir './env/data' --model_dir './vae/model' --epochs 20
```
