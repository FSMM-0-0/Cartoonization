#include "stdafx.h"
#include "cartoon.h"

int g_edge_coreSize = 5;
int g_blur_coreSize = 7;
int g_s = 10;  //���Ͷ�
int g_v = 10; //����

void change_edge(int pos, void*) {
	g_edge_coreSize = pos;
}

void change_blur(int pos, void*) {
	g_blur_coreSize = pos;
}

void change_s(int pos, void*) {
	g_s = pos;
}

void change_v(int pos, void*) {
	g_v = pos;
}

//��Ե��ͼ
void pasteEdge(Mat &image, Mat &outImg, Mat edgeMat)
{
	//��ֵת��
	edgeMat = edgeMat < 100;
	image.copyTo(outImg, edgeMat);
}

//Canny��Ե��⣨�ֱߣ�
Mat Edgedetect_Canny(Mat img)
{
	Mat dstpic, edge, grayImage;
	int threshold = 30;

	//������srcͬ���ͺ�ͬ��С�ľ���
	dstpic.create(img.size(), img.type());
	//��ԭʼͼת��Ϊ�Ҷ�ͼ
	cvtColor(img, grayImage, COLOR_RGB2GRAY);
	//��ʹ��3*3�ں�������
	blur(grayImage, edge, Size(3, 3));
	//����canny����
	Canny(edge, edge, threshold, threshold * 3, 3);

	vector<vector<Point>>g_vContours;
	vector<Vec4i>g_vHierarchy;
	RNG G_RNG(1234);
	findContours(edge, g_vContours, g_vHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat Drawing = Mat::zeros(edge.size(), CV_8UC3);
	for (int i = 0; i < g_vContours.size(); i++) {
		Scalar color = Scalar(255, 255, 255);
		drawContours(Drawing, g_vContours, i, color, 2, 8, g_vHierarchy, 0, Point());
	}

	Mat pasteEdgeCanny;
	pasteEdge(img, pasteEdgeCanny, Drawing);
	return pasteEdgeCanny;
}

//Canny��Ե��⣨ϸ�ߣ�
Mat Edgedetect_Canny2(Mat img)
{
	Mat dstpic, edge, grayImage;
	int threshold = 30;

	dstpic.create(img.size(), img.type());
	cvtColor(img, grayImage, COLOR_RGB2GRAY);
	blur(grayImage, edge, Size(3, 3));
	Canny(edge, edge, threshold, threshold * 3, 3);
	Mat pasteEdgeCanny;
	pasteEdge(img, pasteEdgeCanny, edge);
	return pasteEdgeCanny;
}

//Laplacian��Ե���
Mat Edgedetect_Laplacian(Mat img)
{
	Mat gray, edge;

	//��˹�˲���������
	GaussianBlur(img, img, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//ת��Ϊ�Ҷ�ͼ
	cvtColor(img, gray, CV_BGR2GRAY);
	g_blur_coreSize = g_blur_coreSize / 2 * 2 + 1;
	//��ֵ�˲�
	medianBlur(gray, gray, g_blur_coreSize);
	g_edge_coreSize = g_edge_coreSize / 2 * 2 + 1;
	Laplacian(gray, edge, CV_16S, g_edge_coreSize);
	Mat pasteEdgeLaplacian;
	pasteEdge(img, pasteEdgeLaplacian, edge);
	return pasteEdgeLaplacian;
}

//˫���˲�
Mat Bila_filter(Mat img)
{
	Mat bilaMat;
	bilateralFilter(img, bilaMat, 10, 50, 50, BORDER_DEFAULT);
	return bilaMat;
}

//��HSIת��ΪRGB
Mat HSI2RGB(CvMat* HSI_H, CvMat* HSI_S, CvMat* HSI_I)
{
	IplImage* RGB_Image = cvCreateImage(cvGetSize(HSI_H), IPL_DEPTH_8U, 3);
	double tmpR, tmpG, tmpB, tmpH, tmpS, tmpI;

	for (int i = 0; i < RGB_Image->height; i++) {
		for (int j = 0; j < RGB_Image->width; j++) {
			tmpH = cvmGet(HSI_H, i, j);
			tmpS = cvmGet(HSI_S, i, j);
			tmpI = cvmGet(HSI_I, i, j);

			if (tmpH < 120 && tmpH >= 0) { //RG����
				tmpH = tmpH * PI / 180;
				tmpB = tmpI * (1 - tmpS);
				tmpR = tmpI * (1 + (tmpS * cos(tmpH)) / cos(PI / 3 - tmpH));
				tmpG = (3 * tmpI - (tmpR + tmpB));
			}
			else if (tmpH < 240 && tmpH >= 120) { //GB����
				tmpH -= 120;
				tmpH = tmpH * PI / 180;
				tmpR = tmpI * (1 - tmpS);
				tmpG = tmpI * (1 + tmpS * cos(tmpH) / cos(PI / 3 - tmpH));
				tmpB = (3 * tmpI - (tmpR + tmpG));
			}
			else { //BR����
				tmpH -= 240;
				tmpH = tmpH * PI / 180;
				tmpG = tmpI * (1 - tmpS);
				tmpB = tmpI * (1 + (tmpS * cos(tmpH)) / cos(PI / 3 - tmpH));
				tmpR = (3 * tmpI - (tmpG + tmpB));
			}

			cvSet2D(RGB_Image, i, j, cvScalar(tmpB * 255, tmpG * 255, tmpR * 255));
		}
	}

	return cvarrToMat(RGB_Image);
}

//���Ͷȸ�
Mat Color_HSI(Mat img)
{
	Mat outImg;
	float s_val = 5; //���Ͷ�
	float h_val = 1; //ɫ��
	float i_val = 1; //����
	int row = img.rows;
	int col = img.cols;

	//HSI�ռ����ݾ���
	CvMat* HSI_H = cvCreateMat(row, col, CV_32FC1);
	CvMat* HSI_S = cvCreateMat(row, col, CV_32FC1);
	CvMat* HSI_I = cvCreateMat(row, col, CV_32FC1);

	uchar* ptr_data; //ԭͼ����ָ�룬HSI��������ָ��
	int img_r, img_g, img_b; //rgb����
	int min_rgb; 
	float Hue, Saturation, Intensity; //HSI����
	int channels = img.channels();

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			//�õ�ÿ�����rgb
			ptr_data = img.ptr<uchar>(i);
			ptr_data += j * channels;
			img_b = *ptr_data; ptr_data++;
			img_g = *ptr_data; ptr_data++;
			img_r = *ptr_data;
			min_rgb = min(img_r, img_g);
			min_rgb = min(min_rgb, img_b);

			//���Ͷȷ���[0,1] S
			Saturation = 1 - (float)(3 * min_rgb) / (img_r + img_g + img_b);
			
			//���ȷ���[0,1] I
			Intensity = (float)((img_b + img_g + img_r) / 3) / 255;

			//ɫ������ H
			float numerator = (img_r - img_g + img_r - img_b) / 2;
			float denominator = sqrt(pow((img_r - img_g), 2) + (img_r - img_b) * (img_g - img_b));
			if (denominator != 0) {
				float theta = acos(numerator / denominator) * 180 / PI;
				img_b <= img_g ? Hue = theta : Hue = 360 - theta;
			}
			else
				Hue = 0;

			cvmSet(HSI_H, i, j, Hue * h_val);
			cvmSet(HSI_S, i, j, Saturation * s_val);
			cvmSet(HSI_I, i, j, Intensity * i_val);
		}
	}

	outImg = HSI2RGB(HSI_H, HSI_S, HSI_I);

	//imshow("��ɫ", outImg);
	return outImg;
}

//���Ͷȵ�
Mat Color(Mat img)
{
	Mat hsv;
	cvtColor(img, hsv, CV_BGR2HSV);
	vector<Mat> channels;
	split(hsv, channels);
	channels[1] += g_s;
	channels[2] += g_v;
	merge(channels, hsv);
	cvtColor(hsv, hsv, CV_HSV2BGR);
	return hsv;
}

//��ɫ�ӳ�
Mat Color_R(Mat img)
{
	Mat hsv;
	cvtColor(img, hsv, CV_BGR2HSV);
	vector<Mat> channels;
	split(hsv, channels);
	channels[0] += 100;
	channels[1] += g_s;
	channels[2] += g_v;
	merge(channels, hsv);
	cvtColor(hsv, hsv, CV_HSV2BGR);
	return hsv;
}

//�������
bool Face_Detect(Mat img)
{
	string face_xml = "C:\\OpenCV\\opencv\\sources\\data\\haarcascades_cuda\\haarcascade_frontalface_default.xml";
	CascadeClassifier face;
	if (!face.load(face_xml)) {
		cout << "xml�ļ����ش���" << endl;
		return false;
	}
	vector<Rect> v_face;
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	equalizeHist(gray, gray);
	face.detectMultiScale(gray, v_face, 1.1, 3, 0, Size(50, 50), Size(500, 500));
	for (vector<Rect>::const_iterator iter = v_face.begin(); iter != v_face.end(); iter++) {
		return true; //��⵽����
	}
	return false;
}

//�Զ�������
void createBar(String winName)
{
	createTrackbar("��Ե��ϸ", winName, &g_edge_coreSize, 10, change_edge);
	createTrackbar("��Ե���", winName, &g_blur_coreSize, 10, change_blur);
	createTrackbar("���Ͷ�", winName, &g_s, 255, change_s);
	createTrackbar("����", winName, &g_v, 255, change_v);
}

//���ݲ�ͬ���Ϳ�ͨ��
Mat Cartoon(Mat img, int type)
{
	if (Face_Detect(img)) { //����ͼ
		img = Bila_filter(img);
		img = Color(img);
		img = Edgedetect_Laplacian(img);
	}
	else { //������ͼ
		if (type == 0) { 
			img = Bila_filter(img);
			img = Color_HSI(img);
			img = Edgedetect_Laplacian(img);
		}
		else if (type == 1) {
			img = Bila_filter(img);
			img = Color(img);
			img = Edgedetect_Laplacian(img);
		}
		else if (type == 2) {
			img = Bila_filter(img);
			img = Color_HSI(img);
			img = Edgedetect_Canny(img);
		}
		else if (type == 3) {
			img = Bila_filter(img);
			img = Color_HSI(img);
			img = Edgedetect_Canny2(img);
		}
		else if (type == 4) {
			img = Bila_filter(img);
			img = Color_R(img);
			img = Edgedetect_Canny2(img);
		}
	}

	return img;
}
