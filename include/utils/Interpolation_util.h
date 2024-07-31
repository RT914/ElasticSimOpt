#ifndef __INTERPOLATION_UTIL_H__
#define __INTERPOLATION_UTIL_H__

double HatFunction(double x);
double DifferentialHatFunction(double x);
int GridToFlat(Eigen::Vector3i grid_index);
Eigen::Vector3i FlatToGrid(int flat_index);
double RiemannSum1(const Eigen::Vector3i& v, double h);
double RiemannSum3(const Eigen::Vector3i& v, const Eigen::Vector3i& grid_xi, const Eigen::Vector3d& theta, double h);
double RiemannSum4(const Eigen::Vector3i& grid_xi, const Eigen::Vector3d& theta, double h);
double RiemannSum5(const Eigen::MatrixXi& m, const Eigen::VectorXi& axis, double h);
double RiemannSum6(const Eigen::MatrixXi& m, const Eigen::VectorXi& axis, double h);
double RiemannSum7(const Eigen::MatrixXi& m, const Eigen::VectorXi& axis, double h);

double RiemannSumForDetF(const Eigen::VectorXd& phi, const Eigen::Vector3i& grid_xi, double h);

// check grid diff
bool allElementsWithinOne(const Eigen::Vector3i& vec);
bool allElementsWithinTwo(const std::vector<Eigen::Vector3i>& diff_vectors);

#endif
