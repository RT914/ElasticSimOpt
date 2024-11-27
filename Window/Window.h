#ifndef __WINDOW_H__
#define __WINDOW_H__

#include "../Camera/Camera.h"

const int WindowPositionX = 100;  //生成するウィンドウ位置のX座標
const int WindowPositionY = 100;  //生成するウィンドウ位置のY座標
const int WindowWidth = 768;    //生成するウィンドウの幅
const int WindowHeight = 768;    //生成するウィンドウの高さ
const char WindowTitle[] = "ElasticSimOpt";  //ウィンドウのタイトル

void projection_and_modelview(const Camera& in_Camera);

#endif