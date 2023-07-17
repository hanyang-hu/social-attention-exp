# Social Attention Experiment

An implementation of the paper [Social Attention for Autonomous Decision-Making in Dense Traffic](https://arxiv.org/pdf/1911.12250.pdf) with custom modification (experiments with SAC and GAIL).

We use the low-level continuous action so that it might be harder for the agent to learn from scratch (RGB rendering shows that the agent goes crazy and few other vehicles appeared in the view, probably the attention module is not working at all), perhaps next step is to try out SAC over discrete action space (with the DiscreteMetaAction type).

Simply install the requirements and run the following command, enjoy (though most likely not)~~ 

```
python train.py
```

## To-do List

- [x] Try out [DiscreteMetaAction](http://highway-env.farama.org/actions/#discrete-meta-actions) (which clearly avoids the problem from the agent to go crazy, and we put rgn_render on so that you can actually see how it performs.)

- [ ] Treat a trained agent as expert (or maybe collect manual control data) and try out [GAIL](https://arxiv.org/pdf/1606.03476.pdf) 

- [ ] Incorporate DRQN to enhance performance (maybe refer to this [Playing FPS Games with Deep Reinforcement Learning](https://arxiv.org/pdf/1609.05521.pdf))

## Results

### Continuous Action (3000 episodes, reward scale x10)

![image](./result01.png)


