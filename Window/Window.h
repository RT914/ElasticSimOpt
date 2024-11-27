#ifndef __WINDOW_H__
#define __WINDOW_H__

#include "../Camera/Camera.h"

const int WindowPositionX = 100;  //��������E�B���h�E�ʒu��X���W
const int WindowPositionY = 100;  //��������E�B���h�E�ʒu��Y���W
const int WindowWidth = 768;    //��������E�B���h�E�̕�
const int WindowHeight = 768;    //��������E�B���h�E�̍���
const char WindowTitle[] = "ElasticSimOpt";  //�E�B���h�E�̃^�C�g��

void projection_and_modelview(const Camera& in_Camera);

#endif