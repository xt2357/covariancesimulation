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



pair<double, double> pair_mean(const vector<pair<double, double>> &data) {
	pair<double, double> ans(0.0, 0.0);
	for (auto &d : data) {
		ans.first += d.first;
		ans.second += d.second;
	}
	ans.first /= data.size();
	ans.second /= data.size();
	return ans;
}

double cov(const vector<pair<double, double>> &data) {
	auto avg = pair_mean(data);
	double ans = 0.0;
	for (auto d : data) {
		ans += (d.first - avg.first) * (d.second - avg.second);
	}
	return ans / (data.size() - 1);
}

// estimate cov(Mt, Mtt)
// bucketing_result[i] is the bucket that user i is assigned to 
double bucket_based_cov_estimate(const MatrixXd &samples, const int N, const int bucket_size, const vector<int> &bucketing_result, const int t, const int tt) {
    // Mt: s1/s2, Mtt: s3/s4
	vector<double> s1(bucket_size),s2(bucket_size),s3(bucket_size),s4(bucket_size);
	for (int i = 0; i < N; ++i) {
        auto b = bucketing_result[i];
		s1[b] += samples(t, i);
		s2[b] += 1;
		s3[b] += samples(tt, i);
		s4[b] += 1;
	}
    vector<pair<double, double>> s13, s24, s14, s23;
	for (int i = 0; i < bucket_size; ++i) {
		s13.push_back({s1[i], s3[i]});
		s24.push_back({s2[i], s4[i]});
		s14.push_back({s1[i], s4[i]});
		s23.push_back({s2[i], s3[i]});
	}
	auto avg_13 = pair_mean(s13);
	auto avg_24 = pair_mean(s24);
	return (1.0) * bucket_size * (
		cov(s13)/(avg_24.first*bucket_size)/(avg_24.second*bucket_size)+
		cov(s24)*(avg_13.first*bucket_size)*(avg_13.second*bucket_size)/pow(avg_24.first*bucket_size*avg_24.second*bucket_size, 2.0)-
		cov(s14)*(avg_13.second*bucket_size)/(avg_24.first*bucket_size)/pow(avg_24.second*bucket_size, 2.0)-
		cov(s23)*(avg_13.first*bucket_size)/pow(avg_24.first*bucket_size, 2.0)/(avg_24.second*bucket_size)
	);  
}

// result(i,j) is the estimation of cov(Mi,Mj)
void calc_bucket_based_cor_matrix(const MatrixXd &samples, const int T, const int N, const int bucket_size, MatrixXd &result) {
    std::uniform_int_distribution<int> uniform(0,bucket_size-1);
    vector<int> bucketing_result(N, -1);
    for (int i = 0; i < N; ++i) {
        int b = uniform(Generator::Get());
        bucketing_result[i] = b;
    }
    for (int t = 0; t < T; ++t) {
        for (int tt = 0; tt <= t; ++tt) {
            double estimated_cov = bucket_based_cov_estimate(samples, N, bucket_size, bucketing_result, t, tt);
            result(t, tt) = estimated_cov;
            result(tt, t) = estimated_cov;
        }
    }
}

void multivariate_normal_simulation(bool early_stopping=false, bool path_likelihood=false, bool use_estimated_cor=false) {
    // number of periods
    const int kT = 30;
    // number of users
    const int kN = 4000;
    // number of buckets
    const int kBucketSize = kT*10;
    const int kRuns = 20000;
    // probability of H1 to be true
    const double kP = 0.5;
    const double kDelta = 0.3;
    // stopping criterion
    const double kK = 9.0;
    double discovery = 0, false_discovery = 0;
    double h1_cnt = 0, h0_cnt = 0;

    cout << " T: " << kT 
         << " N: " << kN 
         << " BucketSize: " << kBucketSize 
         << " Runs: " << kRuns 
         << " P: " << kP 
         << " Delta: " << kDelta
         << " K: " << kK
         << endl; 

    // construct a invertible transformation A and make Samples = AX+delta
    // where X consist of i.i.d standard normal random variables
    MatrixXd delta = MatrixXd::Ones(kT, kN) * kDelta;
    MatrixXd A = MatrixXd::Zero(kT, kT);
    for (int i = 0; i < kT; ++i) {
        double factor = 15.0; 
        for (int j = 0; j < i; ++j) {
            A(i,j) = factor/(i);
            // A(i,j) = 0;
        }
        A(i, i) = 30.0;
    }
    // with A we can generate multivariate normal samples, the samples of N users in T periods are represented as a T*N matrix
    // every col of samples matrix(T*N matrix) is a multivariate normal sample of size T, representing the outcome of one user in T periods
    // there are N cols in samples matrix, which represents N users
    // for each period t we have N outcomes of each user, and we can calculate mean of this outcomes, namely Mt
    // covariance matrix of means in each period
    MatrixXd sigma = A*A.transpose()*(1.0/kN);
    // cout << "real Sigma: " << endl << sigma << endl;
    // variance of means
    vector<double> variances(kT, 0);
    for (int i = 0; i < kT; ++i) {
        variances[i] = sigma(i,i);
    }
    //sub_sigma[t] is the sigma related to (M1,M2,...,Mt)
    std::vector<MatrixXd,aligned_allocator<MatrixXd>> sub_sigmas;
    std::vector<MatrixXd,aligned_allocator<MatrixXd>> sub_deltas;
    for (int i = 0; i < kT; ++i) {
        sub_sigmas.push_back(sigma.block(0, 0, i+1, i+1));
        sub_deltas.push_back(delta.block(0,0, i+1,1));
    }
    std::vector<MatrixXd,aligned_allocator<MatrixXd>> inverse_sub_sigmas;
    for (int i = 0; i < kT; ++i) {
        inverse_sub_sigmas.push_back(sub_sigmas[i].inverse());
    }
    std::uniform_real_distribution<double> uni(0.0,1.0);
    for (int r = 0; r < kRuns; ++r) {
        auto dice = uni(Generator::Get());
        int h = dice <= kP ? 1 : 0;
        std::normal_distribution<double> norm(0, 1.0);
        
        MatrixXd samples = MatrixXd::Zero(kT, kN);
        for (int i = 0; i < kT; ++i) {
            for (int j = 0; j < kN; ++j) {
                samples(i,j) = norm(Generator::Get());
            }
        }
        if (h) {
            samples = A*samples + delta;
            h1_cnt += 1;
        }
        else {
            samples = A*samples;
            h0_cnt += 1;
        }

        MatrixXd estimated_sigma = MatrixXd::Zero(kT, kT);
        calc_bucket_based_cor_matrix(samples, kT, kN, kBucketSize, estimated_sigma);
        // cout << "estimated Sigma: " << endl << estimated_sigma << endl;
        std::vector<MatrixXd,aligned_allocator<MatrixXd>> inverse_estimated_sub_sigmas;
        for (int i = 0; i < kT; ++i) {
            inverse_estimated_sub_sigmas.push_back(estimated_sigma.block(0, 0, i+1, i+1).inverse());
        }

        MatrixXd means = MatrixXd::Zero(kT, 1);
        for (int i = 0; i < kT; ++i) {
            means(i,0) = samples.row(i).mean();
        }
        double bf = 1.0, path_bf = 1.0, current_bf = 1.0;
        for (int i = 0; i < kT; ++i) {
            double mean = means(i, 0);
            double var = variances[i];
            bf *= std::exp(-(mean-kDelta)*(mean-kDelta)/2.0/var - -(mean)*(mean)/2.0/var);
            if (path_likelihood) {
                MatrixXd sub_means = means.block(0,0,i+1,1);
                MatrixXd &this_inverse_sub_sigma = use_estimated_cor ? inverse_estimated_sub_sigmas[i] : inverse_sub_sigmas[i];
                double h1_likelihood = ((sub_means-sub_deltas[i]).transpose()* (this_inverse_sub_sigma)*(sub_means-sub_deltas[i]))(0,0);
                double h0_likelihood = ((sub_means).transpose()* (this_inverse_sub_sigma)*(sub_means))(0,0);
                // cout << "likelihoods: " << h1_likelihood << " " << h0_likelihood << endl;
                path_bf = std::exp(-0.5*h1_likelihood - -0.5*h0_likelihood);
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
            std::string str3 = use_estimated_cor ? "true" : "false";
            std::cout << " early_stop=" << str << ", path_likelihood=" << str2 << ", use_estimated_cov=" << str3
                      << " FDR: " << false_discovery << "/" << discovery << "=" << false_discovery/discovery 
                      << " Power: " << discovery-false_discovery <<"/" << h1_cnt << "=" << (discovery-false_discovery)/h1_cnt
                      << endl;
        }
    }
    cout << endl;
}



int main() {
    multivariate_normal_simulation(true,false,false);
    multivariate_normal_simulation(true,true,false);

    multivariate_normal_simulation(true,true,true);
    return 0;
}