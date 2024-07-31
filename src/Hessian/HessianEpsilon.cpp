#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"

void recursiveLoopForHesEpsilon(int level, int maxLevel, const std::function<void(std::vector<int>&)>& process, std::vector<int>& indices) {
    if (level == maxLevel) {
        process(indices);
        return;
    }

    for (int i = 0; i < NumberOfParticles; i++) {
        indices[level] = i;
        recursiveLoopForHesEpsilon(level + 1, maxLevel, process, indices);
    }
}

// Caluculate HessianXi
Eigen::MatrixXd calHessianEpsilon(const Square& square, const Eigen::VectorXd& phi)
{
    Eigen::MatrixXd HessianEpsilon = Eigen::MatrixXd::Zero(3 * NumberOfParticles, 3 * NumberOfParticles);

    auto processIndices = [&](const std::vector<int>& indices) {
        int xi = indices[0], i = indices[1];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);
        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;

        if (allElementsWithinOne(i_minus_xi)) {
            // êœï™åvéZ
            double W = RiemannSum1(i_minus_xi, square.dx);

            // çXêV
            HessianEpsilon.block<3, 3>(3 * i, 3 * xi).diagonal().array() += rho * W;
        }
        };

    std::vector<int> indices(2);
    recursiveLoopForHesEpsilon(0, 2, processIndices, indices);

    return HessianEpsilon;
}
