## Some potentially useful links for pytorch and gym
- For PyTorch:  
	- https://www.cs.utexas.edu/~yukez/cs391r_fall2021/slides/tutorial_09-29_pytorch_intro.pdf#page=6.00
	- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
	- https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/slides/lec-3.pdf#page=25.00
	- https://cs230.stanford.edu/blog/pytorch/
- For Gymnasium:  
	- https://www.gocoder.one/blog/rl-tutorial-with-openai-gym/
	- https://gymnasium.farama.org/introduction/basic_usage/
	- https://blog.paperspace.com/getting-started-with-openai-gym/

## Setup

You can run this code on your own machine, on Google Colab, or on RCAC Scholar. We recommend Python 3.8+.  

You will need to install some Python packages to run this code. Refer to the installation instructions in the homework PDF.

## Complete the code

Fill in sections marked with `TODO`.
See the homework PDF for details.

## Run the code

### Section 1 (Behavior Cloning)
Command for Question 1:

```
python main.py --env CartPole-v1
python main.py --env Blackjack-v1
```

To change hyperparameter values:

```
python main.py --env CartPole-v1 --lr=1e-5 --batch_size=64 --num_epochs=50
```

See the homework PDF for more details on what else you need to run.


## Visualization:

You can visualize your runs using tensorboard:

```
tensorboard --logdir runs
```

Or if this fails to work for some reason, you can use:

```
python -m tensorboard.main --logdir runs
```

You will see scalar summaries as well as videos of your trained policies (in the 'images' tab).

You can choose to visualize specific runs with a comma-separated list:
```
tensorboard --logdir runs/run1,runs/run2,runs/run3...
```