from bot_2d_rep import *

import numpy as np
import copy

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import Sampling, FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize

class SensorPkgOptimization(ElementwiseProblem):

    def __init__(self, bot:SimpleBot2d, sensor_options:list[FOV2D|None], max_n_sensors:int=10, ):
        """
        Initializes the sensor package optimization problem.

        Design Variables (each of N sensors):
            type :      int (sensor object enumerated)
            x :         float (meters)
            y :         float (meters)
            rotation :  float (0-360 deg)
        """
        self.bot = bot
        if None not in sensor_options:
            sensor_options.insert(0, None)
        self.sensor_options = dict(enumerate(sensor_options))
        self.max_n_sensors = max_n_sensors
        s_bounds = np.array([constraint.bounds for constraint in bot.sensor_pose_constraint])
        x_bounds = (np.min(s_bounds[:, 0]), np.max(s_bounds[:, 2]))
        y_bounds = (np.min(s_bounds[:, 1]), np.max(s_bounds[:, 3]))

        variables = dict()
        for i in range(max_n_sensors):
            variables[f"s{i}_type"] = Integer(bounds=(0,len(self.sensor_options)-1))
            variables[f"s{i}_x"] = Real(bounds=(x_bounds[0], x_bounds[1]))
            variables[f"s{i}_y"] = Real(bounds=(y_bounds[0], y_bounds[1]))
            variables[f"s{i}_rotation"] = Real(bounds=(0.0, 360.0))
        
        super().__init__(vars=variables, n_obj=2, n_eq_constr=1)

    def convert_sensor_to_1D(self, sensor:FOV2D, idx:int):
        def get_sensor_key(sensor):
            for key, s in self.sensor_options.items():
                if s == sensor:
                    return key
            raise KeyError
        
        if sensor is not None:
            return {
                f"s{idx}_type": get_sensor_key(sensor),
                f"s{idx}_x": sensor.focal_point[0],
                f"s{idx}_y":sensor.focal_point[1],
                f"s{idx}_rotation":sensor.rotation
            }
        else:
            return {
                f"s{idx}_type": 0,
                f"s{idx}_x": 0,
                f"s{idx}_y": 0,
                f"s{idx}_rotation": 0
            }
    
    def convert_1D_to_sensor(self, x:dict, idx:int):

        if self.sensor_options[x[f"s{idx}_type"]] is None:
            return None
        else:
            sensor = copy.deepcopy(self.sensor_options[x[f"s{idx}_type"]])
            sensor.set_translation(x[f"s{idx}_x"], x[f"s{idx}_y"])
            sensor.set_rotation(x[f"s{idx}_rotation"])
        return sensor

    def convert_bot_to_1D(self, bot):
        x = {}
        for i, sensor in enumerate(bot.sensors):
            sensor_dict = self.convert_sensor_to_1D(sensor, i)
            x.update(sensor_dict)
        x = np.array(list(x.values()))
        return x

    def convert_1D_to_bot(self, x):
        print("Convert 1d->bot X:", x)
        xs = []
        for i in range(self.max_n_sensors):
            sensor_dict = {k: v for k, v in x.items() if k.startswith(f"s{i}_")}
            xs.append(sensor_dict)
        bot = copy.deepcopy(self.bot)
        print("Convert 1d->bot X split:", xs)
        bot.sensors = [self.convert_1D_to_sensor(x, i) for i, x in enumerate(xs)]
        return bot
        

    def _evaluate(self, x, out, *args, **kwargs):
        print("In EVALUATE:", x)
        bot = self.convert_1D_to_bot(x)
        out["F"] = [
            1 - bot.get_sensor_coverage(),  # maximize sensor coverage, so subtract from 1
            bot.get_pkg_cost()              # minimize cost as is
            ]
        out["H"] = bot.is_valid_pkg()
    
class CustomSensorPkgRandomSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        xl = list(xl.values())
        xu = list(xu.values())
        assert np.all(xu >= xl)

        bot = copy.deepcopy(problem.bot)
        bot.sensors = []

        X = []
        for _ in range(n_samples):
            for i in range(problem.max_n_sensors):
                sensor = problem.convert_1D_to_sensor({
                    f"s{i}_type": np.random.randint(0, len(problem.sensor_options)),
                    f"s{i}_x": 0,
                    f"s{i}_y": 0,
                    f"s{i}_rotation": 0
                }, i)
                if sensor is not None:
                    bot.add_sensor_valid_pose(sensor)
            X.append(problem.convert_bot_to_1D(bot))

        print("X:",X)
        return X
        