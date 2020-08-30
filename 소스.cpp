#include "opencv2/opencv.hpp"
#include<iostream>

using namespace cv;
using namespace std;

Mat hue;

int main(int argc, char* argv[])
{
	Mat img = imread("model_face.jpg"); //모델 영상
	Mat Inputimg = imread("veteran.jpg"); //입력 영상
	namedWindow("Model Image");
	imshow("Model Image", img);
	namedWindow("Input Image");
	imshow("Input Image", Inputimg);


	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV);  //모델 영상 컨버트 HSV

	Mat hsvImage;
	cvtColor(Inputimg, hsvImage, COLOR_BGR2HSV); //입력 영상 컨버트 HSV


	hue.create(hsv.size(), hsv.depth());
	int ch[] = { 0,0 };
	mixChannels(&hsv, 1, &hue, 1, ch, 1); //채널 분리 색(hue)성분 이용

	int histSize =256;
	float hue_range[] = { 0,256 };
	const float* ranges = { hue_range };
	int channels = 0;
	int dims = 1;  //히스토그램 생성 인자들

	Mat hist;
	calcHist(&hue, 1, &channels,Mat(), hist, dims, &histSize, &ranges, true, false); //모델 영상 히스토그램
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat()); //정규화

	Mat backproj;
	calcBackProject(&hsvImage, 1, &channels, hist, backproj, &ranges, 1, true); //입력영상 히스토그램 역투영

	Mat backproj2;
	normalize(backproj, backproj2, 0, 255, NORM_MINMAX, CV_8U); //정규화
	imshow("BackProj2", backproj2); //출력


	waitKey(0);
	return 0;
}
