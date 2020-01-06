# world-models-ppo

[World Model](https://arxiv.org/abs/1803.10122) implementation with [PPO](https://arxiv.org/abs/1707.06347) in PyTorch. This repository builds on [world-models](https://github.com/ctallec/world-models) for the VAE and MDN-RNN implementations and [firedup](https://github.com/kashif/firedup) for the PPO optimization of the Controller network. Check the [firedup setup file](https://github.com/kashif/firedup/blob/master/setup.py) for requirements.

First save a number of the *CarRacing-v0* Gym environment rollouts used for the train and test sets in the ```data_dir``` folder:

```bash
python env/carracing.py --data_dir './env/data' ---n_fold_train 20 ---n_fold_test 1
```

Then train the [Variational Autoencoder](https://arxiv.org/abs/1312.6114) (VAE) using the stored rollouts:

```python
from vae.train import run
run(data_dir='./env/data', vae_dir='./vae/model', epochs=5)
```

Using the pretrained VAE, we train the Recurrent [Mixture Density Network](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) (MDN-RNN) model to predict the future latent state:

```python
from mdnrnn.train import run
run(data_dir='./env/data', vae_dir='./vae/model', mdnrnn_dir='./mdnrnn/model', epochs=5)
```

We can finally train the Controller network which steers the car with PPO:

```python
from rl.algos.ppo.ppo import run
run(exp_name='carracing_ppo', epochs=100)
```
