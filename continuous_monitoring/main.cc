#include <deque>
#include <iostream>
#include <vector>
#include <list>
#include <thread>

#include <cstdio>
#include <cmath>

#include "Eigen/Dense"
#include "Eigen/StdVector"

#include "generator.h"

using namespace std;
using namespace Eigen;


void multivariate_normal_simulation(bool early_stopping=false, bool path_likelihood=false) {
    // number of periods
    const int kT = 100;
    // number of users
    const int kN = 1000;
    const int kRuns = 100000;
    const double kP = 0.5, kDelta = 0.2;
    const double kK = 9.0;
    double discovery = 0, false_discovery = 0;
    double h1_cnt = 0, h0_cnt = 0;

    // construct a invertible transformation to Yi: Y = AX(+delta) 
    // where X is i.i.d standard normal random variables
    MatrixXd delta = MatrixXd::Ones(kT, 1) * kDelta;
    MatrixXd A = MatrixXd::Zero(kT, kT);
    for (int i = 0; i < kT; ++i) {
        double factor = -0.5; 
        for (int j = 0; j < i; ++j) {
            A(i,j) = factor/(i);
        }
        A(i, i) = 1.0;
    }
    // covariance matrix of Y
    MatrixXd sigma = A*A.transpose();
    // variance of \sum{Yi}/i
    vector<double> variances(kT, 0);
    for (int i = 0; i < kT; ++i) {
        variances[i] = sigma.block(0, 0, i+1, i+1).mean();
    }
    // sub transformations related to Yi that i<=k, k from 1 to kT
    std::vector<MatrixXd,aligned_allocator<MatrixXd>> sub_a;
    std::vector<MatrixXd,aligned_allocator<MatrixXd>> sub_delta;
    for (int i = 0; i < kT; ++i) {
        sub_a.push_back(A.block(0, 0, i+1, i+1));
        sub_delta.push_back(delta.block(0,0, i+1,1));
    }
    std::vector<MatrixXd,aligned_allocator<MatrixXd>> inverse_sub_sigma;
    for (int i = 0; i < kT; ++i) {
        inverse_sub_sigma.push_back((sub_a[i]*sub_a[i].transpose()).inverse());
    }
    std::uniform_real_distribution<double> uni(0.0,1.0);
    for (int r = 0; r < kRuns; ++r) {
        auto dice = uni(Generator::Get());
        int h = dice <= kP ? 1 : 0;
        std::normal_distribution<double> norm(0, 1.0);
        MatrixXd samples = MatrixXd::Zero(kT, 1);
        for (int i = 0; i < kT; ++i) {
            samples(i,0) = norm(Generator::Get());
        }
        if (h) {
            samples = A*samples + delta;
            h1_cnt+=1;
        }
        else {
            samples = A*samples;
            h0_cnt += 1;
        }
        double bf = 0, path_bf = 0, current_bf = 0, sum_y = 0;
        for (int i = 0; i < kT; ++i) {
            sum_y += samples(i, 0);
            double mean = sum_y/(i+1);
            double var = variances[i];
            bf = std::exp(-(mean-kDelta)*(mean-kDelta)/2.0/var)/std::exp(-(mean)*(mean)/2.0/var);
            if (path_likelihood) {
                MatrixXd sub_samples = samples.block(0,0,i+1,1);
                double h1_likelihood = ((sub_samples-sub_delta[i]).transpose()*inverse_sub_sigma[i]*(sub_samples-sub_delta[i]))(0,0);
                double h0_likelihood = ((sub_samples).transpose()*inverse_sub_sigma[i]*(sub_samples))(0,0);
                path_bf = std::exp(-0.5*h1_likelihood)/std::exp(-0.5*h0_likelihood);
            }
            current_bf = path_likelihood ? path_bf : bf;
            if (early_stopping && current_bf >= kK) break;
        }
        if (current_bf >= kK) {
            discovery += 1;
            false_discovery += h == 0 ? 1 : 0;
        }
        if ((r+1) % (kRuns/10) == 0) {
            std::string str = early_stopping ? "true" : "false";
            std::string str2 = path_likelihood ? "true" : "false";
            std::cout << " early_stop=" << str << ", path_likelihood=" << str2
                      << " FDR: " << false_discovery << "/" << discovery << "=" << false_discovery/discovery 
                      << " Power: " << discovery-false_discovery <<"/" << h1_cnt << "=" << (discovery-false_discovery)/h1_cnt
                      << endl;
        }
    }
}



int main() {
    multivariate_normal_simulation(true,false);
    //multivariate_normal_simulation(true,false);
    multivariate_normal_simulation(true,true);
    return 0;
}