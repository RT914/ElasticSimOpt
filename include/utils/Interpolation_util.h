#ifndef __INTERPOLATION_UTIL_H__
#define __INTERPOLATION_UTIL_H__

double HatFunction(double x);
double DifferentialHatFunction(double x);
int GridToFlat(Eigen::Vector3i grid_index);
Eigen::Vector3i FlatToGrid(int flat_index);
Eigen::Vector3d calculateStencilBase(const Eigen::Vector3d& cal_point);
std::vector<int> generateStencil(const Eigen::Vector3d& stencil_base, Eigen::MatrixXi& stencil);
double calRiemannJ(const Eigen::Vector3d& cal_point, const Eigen::Vector3i& grid_xi, const Eigen::VectorXd& re_phi, const Eigen::VectorXd& phi, const int NumberOfParticles, const double exp);
bool allElementsWithinOne(const Eigen::Vector3i& vec); // check grid diff
#endif
