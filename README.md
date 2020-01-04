# world-models-ppo

[World Model](https://arxiv.org/abs/1803.10122) implementation with [PPO](https://arxiv.org/abs/1707.06347) in PyTorch. This repository builds on [world-models](https://github.com/ctallec/world-models) for the VAE and MDN-RNN implementations and [firedup](https://github.com/kashif/firedup) for the PPO optimization of the Controller network.

First save a number of the *CarRacing-v0* Gym environment rollouts used for the train and test sets in the ```filepath``` folder:

```bash
python env/carracing.py --filepath './env/data' ---n_fold_train 10 ---n_fold_test 1
```

Then train the [Variational Autoencoder](https://arxiv.org/abs/1312.6114) (VAE) using the stored rollouts:

```bash
python vae/train.py --data_dir './env/data' --vae_dir './vae/model' --epochs 20
```

Using the pretrained VAE, we train the Recurrent [Mixture Density Network](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) (MDN-RNN) model to predict the future latent state:

```bash
python mdnrnn/train.py --data_dir './env/data' --vae_dir './vae/model' --mdnrnn_dir './mdnrnn/model' --epochs 20
```
