#include <GL/freeglut.h>
#include <Eigen/Dense>
#include <stdio.h>
#include <iostream>
#include "Draw.h"
#include "../include/Square.h"

//----------------------------------------------------
// �_�̕`��
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
// �����̂̕`��
//----------------------------------------------------
void drawSquare(Square square, float scale_factor)
{
	int N = square.SideNumber + 1;
	float point_size = 3.0;

	// �_�̕`��
	for (int i = 0; i < square.points.size(); i++)
	{
		double pos_x = square.points[i].position(0) * scale_factor;
		double pos_y = square.points[i].position(1) * scale_factor;
		double pos_z = square.points[i].position(2) * scale_factor;
		drawPoint(pos_x, pos_y, pos_z, point_size);
	}

	// X�����̐��̕`��
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

	// Y�����̐��̕`��
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

	// Z�����̐��̕`��
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
// �����̂̕`��i����������Vector�\���j
//----------------------------------------------------
void drawSquareAndVector(Square square, Eigen::VectorXd update_phi, float scale_factor)
{
	int N = square.SideNumber + 1;
	float point_size = 3.0;
	float vector_width = 3.0;
	float vector_length = 3.0;

	// �x�N�g���𐳋K�����܂����H (1: ���K������, 0: ���̂܂�)
	int norm = 0;

	// ���_�Ԃ̐��̕`��Ɠ_�̕`��
	for (int i = 0; i < square.points.size(); i++) {
		// �n�_ (startV)
		Eigen::Vector3d startV = {
			square.points[i].position(0),
			square.points[i].position(1),
			square.points[i].position(2)
		};

		// �I�_ (endV)
		Eigen::Vector3d endV = {
			update_phi(3 * i),
			update_phi(3 * i + 1),
			update_phi(3 * i + 2)
		};

		// �����x�N�g���̌v�Z (�I�_ - �n�_)
		Eigen::Vector3d direction_vector = endV - startV;

		if (norm == 1) {
			// �����x�N�g���̐��K��
			if (direction_vector.norm() != 0) {
				direction_vector.normalize();
			}
			// �x�N�g���̒����𒲐�
			direction_vector *= vector_length;
		}
		else if(norm == 0) {
			direction_vector *= vector_length;
		}
		

		// �X�P�[�����ꂽ�n�_�̍��W
		Eigen::Vector3d scaled_startV = startV * scale_factor;
		Eigen::Vector3d scaled_endV = scaled_startV + direction_vector;

		// �_��`��
		drawPoint(scaled_startV(0), scaled_startV(1), scaled_startV(2), point_size);

		// �x�N�g����`��
		drawVector(scaled_startV, scaled_endV, vector_width);
	}
	
	// X�����̐��̕`��
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

	// Y�����̐��̕`��
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

	// Z�����̐��̕`��
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
// �n�ʂ̕`��
//----------------------------------------------------
void Ground()
{
	double ground_max_x = 300.0;
	double ground_max_y = 300.0;
	glColor3d(0.2, 0.8, 0.8);  // ��n�̐F
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
// �S�̂̕`��
//----------------------------------------------------
void draw()
{
	glColor3d(0, 0, 0);  // ��n�̐F
	Ground();
}