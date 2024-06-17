#ifndef __INTERPOLATION_UTIL_H__
#define __INTERPOLATION_UTIL_H__

double HatFunction(double x);
double DifferentialHatFunction(double x);
int GridToFlat(Eigen::Vector3i grid_index);
Eigen::Vector3i FlatToGrid(int flat_index);
double RiemannSum1(const Eigen::Vector3i& v, double h);
double RiemannSum2(const Eigen::Matrix3i& m, const Eigen::Vector3i& axis, double h);
double RiemannSum3(const Eigen::Vector3i& v, const Eigen::Vector3i& grid_xi, const Eigen::Vector3d& theta, double h);

#endif
