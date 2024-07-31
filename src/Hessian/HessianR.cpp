#include <Eigen/Dense>
#include "../../include/Square.h"
#include "../../include/FEM.h"
#include "../../include/utils/Interpolation_util.h"
#include "../../include/NewtonRaphsonMethod.h"
#include "../../include/Hessian/HessianR.h"

void recursiveLoopForHesR(int depth, int maxDepth, const std::function<void(std::vector<int>&)>& process, std::vector<int>& indices) {
    if (depth == maxDepth) {
        process(indices);
        return;
    }
    for (int i = 0; i < NumberOfParticles; i++) {
        indices[depth] = i;
        recursiveLoopForHesR(depth + 1, maxDepth, process, indices);
    }
}

// Calculate Hessian R
Eigen::MatrixXd calHessianR(const Square& square, const Eigen::VectorXd& theta)
{
    Eigen::MatrixXd HessianR = Eigen::MatrixXd::Zero(NumberOfParticles, NumberOfParticles);

    auto processIndices = [&](const std::vector<int>& indices) {
        int xi = indices[0], i = indices[1];

        Eigen::Vector3i grid_xi = FlatToGrid(xi);
        Eigen::Vector3i i_minus_xi = FlatToGrid(i) - grid_xi;

        if (allElementsWithinOne(i_minus_xi)) {
            HessianR(i, xi) = (kappa / 2.0) * (RiemannSum1(i_minus_xi, square.dx) +
                RiemannSum3(i_minus_xi, grid_xi, theta, square.dx));
        }
        };

    std::vector<int> indices(2);
    recursiveLoopForHesR(0, 2, processIndices, indices);

    return HessianR;
}
