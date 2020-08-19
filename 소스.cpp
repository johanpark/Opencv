#include "opencv2/opencv.hpp"
#include<iostream>

using namespace cv;
using namespace std;

Mat hue;

int main(int argc, char* argv[])
{
	Mat img = imread("C:/Users/pyhan/OneDrive/���� ȭ��/model_face.jpg"); //�� ����
	Mat Inputimg = imread("C:/Users/pyhan/OneDrive/���� ȭ��/veteran.jpg"); //�Է� ����
	namedWindow("Model Image");
	imshow("Model Image", img);
	namedWindow("Input Image");
	imshow("Input Image", Inputimg);


	Mat hsv;
	cvtColor(img, hsv, COLOR_BGR2HSV);  //�� ���� ����Ʈ HSV

	Mat hsvImage;
	cvtColor(Inputimg, hsvImage, COLOR_BGR2HSV); //�Է� ���� ����Ʈ HSV


	hue.create(hsv.size(), hsv.depth());
	int ch[] = { 0,0 };
	mixChannels(&hsv, 1, &hue, 1, ch, 1); //ä�� �и� ��(hue)���� �̿�

	int histSize =256;
	float hue_range[] = { 0,256 };
	const float* ranges = { hue_range };
	int channels = 0;
	int dims = 1;  //������׷� ���� ���ڵ�

	Mat hist;
	calcHist(&hue, 1, &channels,Mat(), hist, dims, &histSize, &ranges, true, false); //�� ���� ������׷�
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat()); //����ȭ

	Mat backproj;
	calcBackProject(&hsvImage, 1, &channels, hist, backproj, &ranges, 1, true); //�Է¿��� ������׷� ������

	Mat backproj2;
	normalize(backproj, backproj2, 0, 255, NORM_MINMAX, CV_8U); //����ȭ
	imshow("BackProj2", backproj2); //���


	waitKey(0);
	return 0;
}
