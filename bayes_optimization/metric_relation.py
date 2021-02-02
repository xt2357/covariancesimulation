#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

from typing import List

class Metric:
    def __init__(self, sums: List[float], cnts: List[float]):
        self.sums = sums
        self.cnts = cnts

    def cnt(self):
        return sum(self.cnts)

    def mean(self):
        return sum(self.sums) / sum(self.cnts)

def cov(list1, list2):
    assert(len(list1)==len(list2))
    n = len(list1)
    return 1.0*n/(n-1)*(
            sum(map(lambda x: x[0]*x[1], zip(list1, list2)))/n - \
                    sum(list1)/n*sum(list2)/n)

def metric_cov(metric1, metric2):
    B = len(metric1.sums)
    mu_d1, mu_d2 = sum(metric1.cnts)/B, sum(metric2.cnts)/B
    mu_n1, mu_n2 = sum(metric1.sums)/B, sum(metric2.sums)/B
    cov_n1_n2 = cov(metric1.sums, metric2.sums)
    cov_d1_d2 = cov(metric1.cnts, metric2.cnts)
    cov_n1_d2 = cov(metric1.sums, metric2.cnts)
    cov_n2_d1 = cov(metric1.cnts, metric2.sums)
    return 1.0/B*(cov_n1_n2/mu_d1/mu_d2 + \
            cov_d1_d2*mu_n1*mu_n2/pow(mu_d1*mu_d2,2.0) - \
            cov_n1_d2*mu_n2/mu_d1/pow(mu_d2,2.0) - \
            cov_n2_d1*mu_n1/mu_d2/pow(mu_d1,2.0))

def metric_std(metric):
    return math.sqrt(metric_cov(metric, metric))

