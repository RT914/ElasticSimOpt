#include <GL/freeglut.h>
#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include "Draw.h"
#include "../include/Square.h"

//----------------------------------------------------
// 点の描画
//----------------------------------------------------
void drawPoint(double x, double y, double z, float size) {
	glPointSize(size);
	glBegin(GL_POINTS);
	glVertex3d(x, y, z);
	glEnd();
}

void drawVector(Eigen::Vector3d StartPoint, Eigen::Vector3d EndPoint, float width) {
	glColor3d(0.0, 0.0, 1.0);
	glLineWidth(width);
	glBegin(GL_LINES);
	glVertex3d(StartPoint(0), StartPoint(1), StartPoint(2));
	glVertex3d(EndPoint(0), EndPoint(1), EndPoint(2));
	glEnd();
}



//----------------------------------------------------
// 立方体の描画
//----------------------------------------------------
void drawSquare(Square square, float scale_factor)
{
	int N = square.SideNumber + 1;
	float point_size = 3.0;

	// 点の描画
	for (int i = 0; i < square.points.size(); i++)
	{
		double pos_x = square.points[i].position(0) * scale_factor;
		double pos_y = square.points[i].position(1) * scale_factor;
		double pos_z = square.points[i].position(2) * scale_factor;
		drawPoint(pos_x, pos_y, pos_z, point_size);
	}

	// X方向の線の描画
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N - 1; k++) {
				int num = i * pow(N, 2) + j * (N)+k;
				glColor3d(0.0, 0.5, 0.0);
				glBegin(GL_LINES);
				glVertex3d(square.points[num].position(0) * scale_factor,
					square.points[num].position(1) * scale_factor,
					square.points[num].position(2) * scale_factor);
				glVertex3d(square.points[num + 1].position(0) * scale_factor,
					square.points[num + 1].position(1) * scale_factor,
					square.points[num + 1].position(2) * scale_factor);
				glEnd();
			}
		}
	}

	// Y方向の線の描画
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N - 1; j++) {
			for (int k = 0; k < N; k++) {
				int num = i * pow(N, 2) + j * (N)+k;
				glColor3d(0.0, 0.5, 0.0);
				glBegin(GL_LINES);
				glVertex3d(square.points[num].position(0) * scale_factor,
					square.points[num].position(1) * scale_factor,
					square.points[num].position(2) * scale_factor);
				glVertex3d(square.points[num + N].position(0) * scale_factor,
					square.points[num + N].position(1) * scale_factor,
					square.points[num + N].position(2) * scale_factor);
				glEnd();
			}
		}
	}

	// Z方向の線の描画
	for (int i = 0; i < N - 1; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				int num = i * pow(N, 2) + j * (N)+k;
				glColor3d(0.0, 0.5, 0.0);
				glBegin(GL_LINES);
				glVertex3d(square.points[num].position(0) * scale_factor,
					square.points[num].position(1) * scale_factor,
					square.points[num].position(2) * scale_factor);
				glVertex3d(square.points[num + pow(N, 2)].position(0) * scale_factor,
					square.points[num + pow(N, 2)].position(1) * scale_factor,
					square.points[num + pow(N, 2)].position(2) * scale_factor);
				glEnd();
			}
		}
	}
};

//----------------------------------------------------
// 立方体の描画（収束方向のVector表示）
//----------------------------------------------------
void drawSquareAndVector(Square square, Eigen::VectorXd update_phi, float scale_factor)
{
	int N = square.SideNumber + 1;
	float point_size = 3.0;
	float vector_width = 3.0;
	float vector_length = 3.0;

	// ベクトルを正規化しますか？ (1: 正規化する, 0: そのまま)
	int norm = 0;

	// 頂点間の線の描画と点の描画
	for (int i = 0; i < square.points.size(); i++) {
		// 始点 (startV)
		Eigen::Vector3d startV = {
			square.points[i].position(0),
			square.points[i].position(1),
			square.points[i].position(2)
		};

		// 終点 (endV)
		Eigen::Vector3d endV = {
			update_phi(3 * i),
			update_phi(3 * i + 1),
			update_phi(3 * i + 2)
		};

		// 方向ベクトルの計算 (終点 - 始点)
		Eigen::Vector3d direction_vector = endV - startV;

		if (norm == 1) {
			// 方向ベクトルの正規化
			if (direction_vector.norm() != 0) {
				direction_vector.normalize();
			}
			// ベクトルの長さを調整
			direction_vector *= vector_length;
		}
		else if(norm == 0) {
			direction_vector *= vector_length;
		}
		

		// スケールされた始点の座標
		Eigen::Vector3d scaled_startV = startV * scale_factor;
		Eigen::Vector3d scaled_endV = scaled_startV + direction_vector;

		// 点を描画
		drawPoint(scaled_startV(0), scaled_startV(1), scaled_startV(2), point_size);

		// ベクトルを描画
		drawVector(scaled_startV, scaled_endV, vector_width);
	}
	
	// X方向の線の描画
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N - 1; k++) {
				int num = i * pow(N, 2) + j * (N)+k;
				glColor3d(0.0, 0.5, 0.0);
				glBegin(GL_LINES);
				glVertex3d(square.points[num].position(0) * scale_factor,
					square.points[num].position(1) * scale_factor,
					square.points[num].position(2) * scale_factor);
				glVertex3d(square.points[num + 1].position(0) * scale_factor,
					square.points[num + 1].position(1) * scale_factor,
					square.points[num + 1].position(2) * scale_factor);
				glEnd();
			}
		}
	}

	// Y方向の線の描画
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N - 1; j++) {
			for (int k = 0; k < N; k++) {
				int num = i * pow(N, 2) + j * (N)+k;
				glColor3d(0.0, 0.5, 0.0);
				glBegin(GL_LINES);
				glVertex3d(square.points[num].position(0) * scale_factor,
					square.points[num].position(1) * scale_factor,
					square.points[num].position(2) * scale_factor);
				glVertex3d(square.points[num + N].position(0) * scale_factor,
					square.points[num + N].position(1) * scale_factor,
					square.points[num + N].position(2) * scale_factor);
				glEnd();
			}
		}
	}

	// Z方向の線の描画
	for (int i = 0; i < N - 1; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				int num = i * pow(N, 2) + j * (N)+k;
				glColor3d(0.0, 0.5, 0.0);
				glBegin(GL_LINES);
				glVertex3d(square.points[num].position(0) * scale_factor,
					square.points[num].position(1) * scale_factor,
					square.points[num].position(2) * scale_factor);
				glVertex3d(square.points[num + pow(N, 2)].position(0) * scale_factor,
					square.points[num + pow(N, 2)].position(1) * scale_factor,
					square.points[num + pow(N, 2)].position(2) * scale_factor);
				glEnd();
			}
		}
	}

};

//----------------------------------------------------
// 地面の描画
//----------------------------------------------------
void Ground()
{
	double ground_max_x = 300.0;
	double ground_max_y = 300.0;
	glColor3d(0.2, 0.8, 0.8);  // 大地の色
	glBegin(GL_LINES);
	for (double ly = -ground_max_y; ly <= ground_max_y; ly += 10.0) {
		glVertex3d(-ground_max_x, ly, 0);
		glVertex3d(ground_max_x, ly, 0);
	}
	for (double lx = -ground_max_x; lx <= ground_max_x; lx += 10.0) {
		glVertex3d(lx, ground_max_y, 0);
		glVertex3d(lx, -ground_max_y, 0);
	}
	glEnd();
}

//----------------------------------------------------
// 全体の描画
//----------------------------------------------------
void draw()
{
	glColor3d(0, 0, 0);  // 大地の色
	Ground();
}