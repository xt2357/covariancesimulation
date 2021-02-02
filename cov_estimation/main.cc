#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <algorithm>
#include <unordered_map>



using namespace std;


default_random_engine &GetGenerator() {
	uint64_t seed = (uint64_t)std::chrono::system_clock::now().time_since_epoch().count();
    seed += (uint64_t)std::hash<std::thread::id>{}(std::this_thread::get_id());
	thread_local static default_random_engine generator(seed);
	return generator;
}

struct User {
	double Y1, Y2, E1, E2;
	int32_t Z1, Z2; 	
};

vector<User> gen_population(int size) {
	/*
		(Y1, Y2) = AX + (1, 2) where X = (X1, X2), X1 and X2 are independent standard normal variables
		thus (Y1, Y2) \sim MultiNormal((100,200), \Sigma) where \Sigma = A*A.transpose()
		A = [
			100,  0
			80, 20
		] 
		thus Y1 = 300X1 and Y2 = 200X1+100X2, and
		\Sigma = [
			10000, 8000
			60000, 6800
		]
	*/
	std::normal_distribution<double> norm(0.0, 1.0);
	vector<User> population(size);
	for (int i = 0; i < size; ++i) {
		double X1 = norm(GetGenerator());
		double X2 = norm(GetGenerator());
		population[i].Y1 = X1 * 100 + 100.0;
		population[i].Y2 = (X1 * 80 + X2 * 20) + 200.0;
	}
	vector<uint32_t> indexes;
	for (int i = 0; i < size; ++i) {
		indexes.push_back(i);
	}
	std::sort(begin(indexes), end(indexes), [&](const int a, const int b) {
		return population[a].Y1 < population[b].Y1;
	});
	for (int i = 0; i < size; ++i) {
		auto idx = indexes[i];
		population[idx].E1 = 1.0*i/size;
	}
	std::sort(begin(indexes), end(indexes), [&](const int a, const int b) {
		return population[a].Y2 < population[b].Y2;
	});
	for (int i = 0; i < size; ++i) {
		auto idx = indexes[i];
		population[idx].E2 = 1.0*i/size;
	}
	for (int i = 0; i < size; ++i) {
		std::bernoulli_distribution d1(1.0 - min(0.5, 1.0-population[i].E1));
		population[i].Z1 = d1(GetGenerator());
		std::bernoulli_distribution d2(1.0 - min(0.5, 1.0-population[i].E2));
		population[i].Z2 = d2(GetGenerator());
	}
	return population;
}

void print_population(const vector<User> &population) {
	for (int i = 0; i < population.size(); ++i) {
		printf(
			"Y1: %.2lf, Y2: %.2lf, Z1: %u, Z2: %u, E1: %.2lf, E2: %.2lf\n", 
			population[i].Y1, population[i].Y2, population[i].Z1, 
			population[i].Z2, population[i].E1, population[i].E2
		);
	}
}


vector<User> sampling(const vector<User> &population, double ratio) {
	vector<User> samples;
	for (int i = 0; i < population.size(); ++i) {
		std::bernoulli_distribution d1(ratio);
		if (d1(GetGenerator())) {
			samples.push_back(population[i]);
		}
	}
	return samples;
}


pair<double, double> calc_metrics(const vector<User> &samples) {
	double sum1 = 0.0, sum2 = 0.0, cnt1 = 0.0, cnt2 = 0.0;
	for (auto &s : samples) {
		sum1 += s.Y1 * s.Z1;
		cnt1 += s.Z1;
		sum2 += s.Y2 * s.Z2;
		cnt2 += s.Z2;
	}
	return {sum1/cnt1, sum2/cnt2};
}

pair<double, double> mean(const vector<pair<double, double>> &data) {
	pair<double, double> ans(0.0, 0.0);
	for (auto &d : data) {
		ans.first += d.first;
		ans.second += d.second;
	}
	ans.first /= data.size();
	ans.second /= data.size();
	return ans;
}

double mean(const vector<double> &data) {
	double sum = 0.0;
	for (auto d : data) {
		sum += d;
	}
	return sum / data.size();
}

double cov(const vector<pair<double, double>> &data) {
	auto avg = mean(data);
	double ans = 0.0;
	for (auto d : data) {
		ans += (d.first - avg.first) * (d.second - avg.second);
	}
	return ans / (data.size() - 1);
}

double var(const vector<double> &data) {
	vector<pair<double, double>> repeat;
	for (auto d : data) {
		repeat.push_back({d, d});
	}
	return cov(repeat);
}

double naive_approach(const vector<User> &samples) {
	double n = 0.0;
	vector<pair<double, double>> obs;
	for (auto &s : samples) {
		if (s.Z1 && s.Z2) {
			obs.push_back({s.Y1, s.Y2});
			n += 1.0;
		}
	}
	return cov(obs) / n;
}

double delta_method_after_data_augmentation(const vector<User> &samples) {
	// metric1: mean_a/mean_b, metric2: mean_c/mean_d
	vector<pair<double, double>> ac,bd,ad,bc;
	double n = 0.0;
	for (auto &s : samples) {
		if (s.Z1 || s.Z2) {
			ac.push_back({s.Y1 * s.Z1, s.Y2 * s.Z2});
			bd.push_back({s.Z1, s.Z2});
			ad.push_back({s.Y1 * s.Z1, s.Z2});
			bc.push_back({s.Z1, s.Y2 * s.Z2});
			n += 1.0;
		}
	}
	auto avg_ac = mean(ac);
	auto avg_bd = mean(bd);
	return (
		cov(ac)/avg_bd.first/avg_bd.second+
		cov(bd)*avg_ac.first*avg_ac.second/pow(avg_bd.first*avg_bd.second, 2.0)-
		cov(ad)*avg_ac.second/avg_bd.first/pow(avg_bd.second, 2.0)-
		cov(bc)*avg_ac.first/pow(avg_bd.first, 2.0)/avg_bd.second
	)/n;
}

double bucket_based(const vector<User> &samples, int bucket_size, double ratio) {
	// metric1: s1/s2, metric2: s3/s4
	vector<double> s1(bucket_size),s2(bucket_size),s3(bucket_size),s4(bucket_size);
	std::uniform_int_distribution<int> uniform(0,bucket_size-1);
	for (auto &s : samples) {
		int b = uniform(GetGenerator());
		s1[b] += s.Y1 * s.Z1;
		s2[b] += s.Z1;
		s3[b] += s.Y2 * s.Z2;
		s4[b] += s.Z2;
	}
	vector<pair<double, double>> s13, s24, s14, s23;
	for (int i = 0; i < bucket_size; ++i) {
		s13.push_back({s1[i], s3[i]});
		s24.push_back({s2[i], s4[i]});
		s14.push_back({s1[i], s4[i]});
		s23.push_back({s2[i], s3[i]});
	}
	auto avg_13 = mean(s13);
	auto avg_24 = mean(s24);
	return (1.0 - ratio) * bucket_size * (
		cov(s13)/(avg_24.first*bucket_size)/(avg_24.second*bucket_size)+
		cov(s24)*(avg_13.first*bucket_size)*(avg_13.second*bucket_size)/pow(avg_24.first*bucket_size*avg_24.second*bucket_size, 2.0)-
		cov(s14)*(avg_13.second*bucket_size)/(avg_24.first*bucket_size)/pow(avg_24.second*bucket_size, 2.0)-
		cov(s23)*(avg_13.first*bucket_size)/pow(avg_24.first*bucket_size, 2.0)/(avg_24.second*bucket_size)
	);
}

void test(int population_size, double ratio, int runs, const vector<int> bucket_sizes) {
	auto population = gen_population(population_size);
	vector<pair<double, double>> metrics;
	vector<double> naive, delta;
	unordered_map<int, vector<double>> bucket_size_2_results;
	for (int r = 0; r < runs; ++r) {
		auto samples = sampling(population, ratio);
		metrics.push_back(calc_metrics(samples));
		naive.push_back(naive_approach(samples));
		delta.push_back(delta_method_after_data_augmentation(samples));
		for (int i = 0; i < bucket_sizes.size(); ++i) {
			bucket_size_2_results[bucket_sizes[i]].push_back(bucket_based(samples, bucket_sizes[i], ratio));
		}
	}
	double ground_truth = cov(metrics);
	printf(
		"ground_truth: %lf\nnaive: %lf, %lf\ndelta: %lf, %lf\n",
		ground_truth, mean(naive), sqrt(var(naive)), mean(delta), sqrt(var(delta))
	);
	for (int bucket_size : bucket_sizes) {
		auto &results = bucket_size_2_results[bucket_size];
		printf("bucket size %d: %lf, %lf\n", bucket_size, mean(results), sqrt(var(results)));
	}
}

int main() {
	test(10000, 0.1, 100000, {100, 200, 500, 1000});
	
	test(10000, 0.2, 100000, {});
	test(10000, 0.1, 100000, {});
	test(10000, 0.05, 100000, {});
	test(10000, 0.01, 100000, {});
	return 0;
}
