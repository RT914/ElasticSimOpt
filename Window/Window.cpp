#include <GL/freeglut.h>

// #include "include/Fem.h"
// #include "Draw/Draw.h"
#include "Window.h"

#define _USE_MATH_DEFINES
#include <math.h>

void projection_and_modelview(const Camera& in_Camera)
{
	const double fovy_deg = (2.0 * 180.0 / M_PI) * atan(0.024 * 0.5 / in_Camera.getFocalLength());

	// 投影行列の設定
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fovy_deg, double(WindowWidth) / double(WindowHeight),
		0.01 * in_Camera.getFocalLength(), 1000.0);

	// モデルビュー行列の設定
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	const Eigen::Vector3d lookAtPoint = in_Camera.getLookAtPoint();

	gluLookAt(in_Camera.getEyePoint().x(), in_Camera.getEyePoint().y(), in_Camera.getEyePoint().z(),
		lookAtPoint.x(), lookAtPoint.y(), lookAtPoint.z(),
		in_Camera.getYVector().x(), in_Camera.getYVector().y(), in_Camera.getYVector().z());
}
