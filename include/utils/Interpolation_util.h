#ifndef __INTERPOLATION_UTIL_H__
#define __INTERPOLATION_UTIL_H__

double HatFunction(double x);
double DifferentialHatFunction(double x);
int GridToFlat(Eigen::Vector3i grid_index);
Eigen::Vector3i FlatToGrid(int flat_index);
double RiemannSumJ(const Eigen::VectorXd& phi, const Eigen::Vector3i& grid_xi, double h, Eigen::VectorXd cal_points, double exp);
double calRiemannJ(const Eigen::Vector3d& cal_point, const Eigen::Vector3i& grid_xi, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, int NumberOfParticles, double exp);
// ëÃêœïœâªó¶åvéZÇÃämîFÇÃÇΩÇﬂÇÃä÷êî
double RiemannSumTest1(double h, double exp);
double RiemannSumTest2(double h);
double RiemannSumTest3(double h, Eigen::VectorXd phi);
double RiemannSumTest4(double h, Eigen::VectorXd phi);

// check grid diff
bool allElementsWithinOne(const Eigen::Vector3i& vec);
#endif
