# pybullet-driving-env
Driving environment designed for asymmetric selfplay

The implementation is based off of [GerardMaggiolino](https://github.com/GerardMaggiolino)'s [simple_driving](https://github.com/GerardMaggiolino/Gym-Medium-Post/tree/main/simple_driving)

Sample usage has been added in `sample-usage.py`

For rendering, use `self.client = p.connect(p.GUI)` instead of `self.client = p.connect(p.DIRECT)` in `__init__()` in `simple_driving_env.py`.