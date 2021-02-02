#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import numpy as np
import os
import pandas as pd
import pdb
import time

from collections import defaultdict
from pandas import Series, DataFrame
from pathlib import Path
from typing import Any, Dict, List, Optional

from ax.core.search_space import SearchSpace
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.simple_experiment import SimpleExperiment
from ax.modelbridge.registry import Models
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.decoder import object_from_json

from functions import get_function_by_name, get_function_id
from metric_relation import Metric, metric_cov

def get_search_space(function_list: List):
    function = function_list[0]
    return SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=function.domain[i][0],
                upper=function.domain[i][1],
            )
            for i in range(function.required_dimensionality)
        ]
    )

def get_groundtruth_function(function_list, weight_list):

    def custom_groundtruth_function(parameterization, weight=None):
        dim = function_list[0].required_dimensionality
        x = np.array([parameterization.get(f"x{i}") for i in range(dim)])
        mean = [function(x) for function in function_list]
        return np.dot(mean, weight_list)

    return custom_groundtruth_function


def get_metrics(mean, cov, num_random, num_bucket):
    new_cov = np.array(cov) * num_random
    samples = np.random.multivariate_normal(mean, new_cov, num_random)

    pd_samples = pd.DataFrame(samples, columns=np.arange(0, samples.shape[1]))
    pd_samples['bid'] = np.random.randint(0, num_bucket, num_random)
    metrics = []
    result = pd_samples.groupby('bid').agg(['sum', 'count'])
    for i in np.arange(0, samples.shape[1]):
        metrics.append(Metric(
            sums=result[i]['sum'].tolist(),
            cnts=result[i]['count'].tolist()))
    return metrics


def get_evaluation_function(
        function_list,
        weight_list,
        covariance_matrix,
        var_compute_type,
        num_random = 10000,
        num_bucket = 50):

    def get_without_var_cov(mean, cov):
        # samples = np.random.multivariate_normal(mean, cov)
        # obj_mean = np.dot(samples, weight_list)
        metrics = get_metrics(mean, cov, num_random, num_bucket)
        means = [metric.mean() for metric in metrics]
        obj_mean = np.dot(means, weight_list)
        return obj_mean, 0.0

    def get_with_groundtruth_var_ignore_cov(mean, cov):

        # samples = np.random.multivariate_normal(mean, cov)
        # obj_mean = np.dot(samples, weight_list)
        metrics = get_metrics(mean, cov, num_random, num_bucket)
        means = [metric.mean() for metric in metrics]
        obj_mean = np.dot(means, weight_list)

        obj_var = 0.0
        for i in range(len(covariance_matrix)):
            obj_var += weight_list[i] * weight_list[i] * covariance_matrix[i][i]
        return obj_mean, obj_var
    
    def get_with_groundtruth_var_cov(mean, cov):

        # samples = np.random.multivariate_normal(mean, cov)
        # obj_mean = np.dot(samples, weight_list)
        metrics = get_metrics(mean, cov, num_random, num_bucket)
        means = [metric.mean() for metric in metrics]
        obj_mean = np.dot(means, weight_list)

        obj_var = 0.0
        for i in range(len(covariance_matrix)):
            for j in range(len(covariance_matrix[i])):
                obj_var += \
                    weight_list[i] * weight_list[j] * covariance_matrix[i][j]
        return obj_mean, obj_var
        
    def get_with_estimate_var_ignore_cov(mean, cov):
        metrics = get_metrics(mean, cov, num_random, num_bucket)
        means = [metric.mean() for metric in metrics]
        obj_mean = np.dot(means, weight_list)
        obj_var = 0.0
        for i in range(len(metrics)):
            obj_var += \
                (weight_list[i] ** 2) * metric_cov(metrics[i], metrics[i])
        return obj_mean, obj_var

    def get_with_estimate_var_cov(mean, cov):
        metrics = get_metrics(mean, cov, num_random, num_bucket)
        means = [metric.mean() for metric in metrics]
        obj_mean = np.dot(means, weight_list)
        obj_var = 0.0
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                mcov = metric_cov(metrics[i], metrics[j])
                obj_var += \
                        weight_list[i] * weight_list[j] * mcov
        return obj_mean, obj_var

    def custom_evaluation_function(parameterization, weight=None):
        dim = function_list[0].required_dimensionality
        x = np.array([parameterization.get(f"x{i}") for i in range(dim)])
        mean = [function(x) for function in function_list]
        cov = covariance_matrix

        obj_mean, obj_var = None, None
        if var_compute_type == 0: # var = 0; covar = 0
            obj_mean, obj_var = get_without_var_cov(mean, cov)
        elif var_compute_type == 1: # var = groundtruth; covar = 0
            obj_mean, obj_var = get_with_groundtruth_var_ignore_cov(mean, cov)
        elif var_compute_type == 2: # var = groundtruth; covar = groundtruth
            obj_mean, obj_var = get_with_groundtruth_var_cov(mean, cov)
        elif var_compute_type == 3: # var = estimate; covar = 0
            obj_mean, obj_var = get_with_estimate_var_ignore_cov(mean, cov)
        elif var_compute_type == 4: # var = estimate; covar = estimate
            obj_mean, obj_var = get_with_estimate_var_cov(mean, cov)
        
        return {"objective_name": (obj_mean, np.sqrt(obj_var))}

    return custom_evaluation_function


# def get_evaluation_function(
#         function_list, weight_list, covariance_matrix, evaluate_covariance, var_coef):
# 
#     def custom_evaluation_function(parameterization, weight=None):
#         dim = function_list[0].required_dimensionality
#         x = np.array([parameterization.get(f"x{i}") for i in range(dim)])
#         mean = [function(x) for function in function_list]
#         cov = covariance_matrix
#         samples = np.random.multivariate_normal(mean, cov)
#         
#         obj_mean = np.dot(samples, weight_list)
#         obj_var = 0.0
#         for i in range(len(covariance_matrix)):
#             for j in range(len(covariance_matrix[i])):
#                 if evaluate_covariance == 0 and i != j:
#                     continue
#                 obj_var += \
#                     weight_list[i] * weight_list[j] * covariance_matrix[i][j]
# 
#         return {"objective_name": (obj_mean, var_coef * np.sqrt(obj_var))}
# 
#     return custom_evaluation_function


def parse_args() -> argparse.Namespace:
    """Parsing command line arguments.

    Returns:
        Parsed argument object.
    """
    parser = argparse.ArgumentParser(description='X sandbox.')
    # 一次可以测试多个函数
    parser.add_argument('-fl', '--function_name_list', type=str,
                        dest='function_name_list', required=True,
                        help='test objective function names list.')
    # 权重列表 for ScalarizedObjective.
    parser.add_argument('-wl', '--weight_list', type=str,
                        dest='weight_list', required=True,
                        help='weight list.')
    # 协方差矩阵
    parser.add_argument('-cm', '--covariance_matrix', type=str,
                        dest='covariance_matrix', required=True,
                        help='covariance matrix.')
    # 是否估计协方差
    parser.add_argument('-ec', '--evaluate_covariance', type=int,
                        dest='evaluate_covariance', default=0,
                        help='0: False; 1: True(evaluate covariance)')
    # # 桶数
    # parser.add_argument('-nb', '--num_bucket', type=int,
    #                     dest='num_bucket', required=False,
    #                     default=1, help='number of bucket.')
    # # 样本量
    # parser.add_argument('-ns', '--num_sample', type=int,
    #                     dest='num_sample', required=False,
    #                     default=1, help='number of sample.')
    # 随机迭代次数
    parser.add_argument('-ii', '--init_iter', type=int,
                        dest='init_iter', required=False,
                        default=1, help='iteration of sobol generation.')
    # 随机迭代时每次迭代的组数
    parser.add_argument('-ibs', '--init_batch_size', type=int,
                        dest='init_batch_size', required=False,
                        default=1, help='number of sobol generation.')
    # BO更新的迭代次数
    parser.add_argument('-ui', '--update_iter', type=int,
                        dest='update_iter', required=False,
                        default=20, help='number of GP(N)EI generation.')
    # 实验组数
    parser.add_argument('-bs', '--batch_size', type=int,
                        dest='batch_size', required=False,
                        default=1, help='number of trial each iter.')
    # 方差系数 deprecated
    parser.add_argument('-vc', '--var_coef', type=int,
                        dest='var_coef', required=False,
                        default=1, help='variance coef.')
    # 方差计算方法
    parser.add_argument('-vct', '--var_compute_type', type=int,
                        dest='var_compute_type', required=False,
                        default=1, help='variance compute type: 0 1 2 3 4')

    # 采样数
    parser.add_argument('-nr', '--num_random', type=int,
                        dest='num_random', required=False,
                        default=10000, help='num_random for gen samples')
    # 桶数
    parser.add_argument('-nb', '--num_bucket', type = int,
                        dest='num_bucket', required=False,
                        default=50, help='number of bucket.')
    # 对照组数
    # parser.add_argument('-nc', '--num_control', type=int,
    #                     dest='num_control', required=False,
    #                     default=0, help='number of control arms each iter.')
    # 保存路径
    parser.add_argument('-sp', '--save_path', type=str, dest='save_path',
        default="/mnt/wfs/mmcommwfssz/project_wx-td-itil-exp/" + \
                "bo_test_output/covariance_test",
        help=("helper directory."))
    return parser.parse_args()

def main():

    args = parse_args()

    function_list = [get_function_by_name[name]
            for name in args.function_name_list.split(',')]
    weight_list = list(map(float, args.weight_list.split(',')))
    covariance_matrix = json.loads(args.covariance_matrix)
    evaluate_covariance = args.evaluate_covariance

    init_iter = args.init_iter
    # if init_iter > 1:
    #     raise ValueError("init_iter should be 1.")
    init_batch_size = args.init_batch_size
    update_iter = args.update_iter
    batch_size = args.batch_size
    var_coef = args.var_coef

    var_compute_type = args.var_compute_type
    num_random = args.num_random
    num_bucket = args.num_bucket

    save_path = args.save_path

    # num_control = args.num_control

    minimize = True

    groundtruth_function = get_groundtruth_function(function_list, weight_list)
    #evaluation_function = get_evaluation_function(
    #        function_list, weight_list, covariance_matrix,
    #        evaluate_covariance, var_coef)
    evaluation_function = get_evaluation_function(
            function_list, weight_list, covariance_matrix,
            var_compute_type, num_random, num_bucket)

    exp = SimpleExperiment(
        name=args.function_name_list + args.weight_list,
        search_space=get_search_space(function_list),
        evaluation_function=evaluation_function,
        objective_name="objective_name",
        minimize=minimize,
    )
    t_start = time.time()
    print(f"Start time: {t_start}")
    print(f"Sobol iteration begin...{time.time() - t_start}")
    sobol = Models.SOBOL(exp.search_space)
    for i in range(init_iter):
        if init_batch_size == 1:
            exp.new_trial(generator_run=sobol.gen(init_batch_size))
        else:
            exp.new_batch_trial(generator_run=sobol.gen(init_batch_size))
        print(f"Running sobol optimization trial {i+1}/{init_iter}..."
              f"{time.time() - t_start}")
    print(f"GPEI iteration begin...{time.time() - t_start}")
    for i in range(update_iter):
        gpei = Models.BOTORCH(experiment=exp, data=exp.eval())
        if batch_size == 1:
            exp.new_trial(generator_run=gpei.gen(batch_size))
        else:
            exp.new_batch_trial(generator_run=gpei.gen(batch_size))
        print(f"Running GPEI optimization trial {i+1}/{update_iter}..."
              f"{time.time() - t_start}")

    # Construct Result.
    ## origin data.
    data_df = copy.deepcopy(exp.eval().df)
    compare_func = min if minimize else max
    
    arm_name2mean = {}
    for _, row in data_df.iterrows():
        arm_name2mean[row["arm_name"]] = row["mean"]
    ## parameters true_mean.
    other_columns = {
        "arm_name": [], "parameters": [], "true_mean": [],
        "cur_trial_best_mean": [], "accum_trials_best_mean": []}
    atbm = None # accum_trial_best_mean
    for trial in exp.trials.values():
        ctbm = None # cur_trial_best_mean
        for arm in trial.arms:
            other_columns['arm_name'].append(arm.name)
            other_columns['parameters'].append(json.dumps(arm.parameters))
            other_columns['true_mean'].append(
                    groundtruth_function(arm.parameters))
            if ctbm is None:
                ctbm = arm_name2mean[arm.name]
            ctbm = compare_func(ctbm, arm_name2mean[arm.name])
        if atbm is None:
            atbm = ctbm
        atbm = compare_func(atbm, ctbm)
        other_columns['cur_trial_best_mean'].extend([ctbm] * len(trial.arms))
        other_columns['accum_trials_best_mean'].extend([atbm] * len(trial.arms))
    other_df = DataFrame(other_columns)

    result_df = data_df.set_index('arm_name').join(
            other_df.set_index('arm_name')).reset_index()
    
    # Save to file.
    print("Save to file.")
    sub_dir_name = "_".join([
        "ax", args.function_name_list.replace(",", "_"),
        args.weight_list.replace(",", "_"), args.covariance_matrix.replace(
            "[", "_").replace("]", "_").replace(",", "_").replace(" ", ""),
        str(args.evaluate_covariance), str(args.init_iter), str(init_batch_size),
        str(args.update_iter), str(args.batch_size), str(args.var_coef),
        str(minimize), str(var_compute_type), str(num_random), str(num_bucket)
        ])
    abs_dir_path = os.path.join(save_path, sub_dir_name)
    Path(abs_dir_path).mkdir(parents=True, exist_ok=True)
    task_id = os.environ.get('TASK_INDEX')
    cur_time = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    filename = cur_time + "_" + str(task_id) + ".csv"
    print(os.path.join(abs_dir_path, filename))
    result_df.to_csv(os.path.join(abs_dir_path, filename))
    print("2021-01-19 19:48:00")
    print("Done...")


if __name__ == '__main__':
    main()
