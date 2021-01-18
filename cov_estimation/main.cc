#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <algorithm>



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
		thus (Y1, Y2) \sim MultiNormal((5,7), \Sigma) where \Sigma = A*A.transpose()
		A = [
			  1,   0
			1/2, 1/2
		] 
		thus Y1 = X1 and Y2 = (X1+X2)/2, and
		\Sigma = [
			  1, 1/2
			1/2, 1/2
		]
	*/
	std::normal_distribution<double> norm(0.0, 1.0);
	vector<User> population(size);
	for (int i = 0; i < size; ++i) {
		double X1 = norm(GetGenerator());
		double X2 = norm(GetGenerator());
		population[i].Y1 = X1 + 3.0;
		population[i].Y2 = (X1 + X2) / 2.0 + 4.0;
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

int main() {
	auto population = gen_population(100);
	for (int i = 0; i < population.size(); ++i) {
		printf(
			"Y1: %.2lf, Y2: %.2lf, Z1: %u, Z2: %u, E1: %.2lf, E2: %.2lf\n", 
			population[i].Y1, population[i].Y2, population[i].Z1, 
			population[i].Z2, population[i].E1, population[i].E2
		);
	}
	return 0;
}
