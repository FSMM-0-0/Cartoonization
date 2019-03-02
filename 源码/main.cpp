#include "stdafx.h"
#include "cartoon.h"
#include "highgui.h"
#include "cv.h"

void Init()
{
	cout << "输入\"1 + picture_path\"选择图片" << endl;
	cout << "输入\"2\"打开摄像头" << endl;
	cout << "输入\"# + 0~4\"实现不同风格的图像卡通化" << endl;
	cout << "点击空格保存图片" << endl;
	cout << "点击ESC关闭退出" << endl;
	system("md .\\save\\");
}

int main()
{
	Init();

	int type, t;
	char c;
	string path;
	string save_path = ".\\save\\";
	Mat img, result;
	char ch[64];

	cin >> t;
	if (t == 1) { //打开单独图像
		cin >> path;
		img = imread(path);
		if (img.empty()) {
			cout << "图片加载失败" << endl;
		}
		else {
			cin >> c >> type;
			result = Cartoon(img, type);
			string str = "卡通化_" + to_string(type);
			imshow(str, result);
			char key = waitKey();

			if (key == 32) {
				time_t t = time(0);
				strftime(ch, sizeof(ch), "%Y-%m-%d %H-%M-%S", localtime(&t));
				imwrite(save_path + ch + ".jpg", result);
			}
		}
	}
	else if (t == 2) { //打开摄像头
		namedWindow("自动调节", 0);
		createBar("自动调节");

		VideoCapture cap(0);
		Mat frame;
		while (cap.isOpened()) {
			if (!cap.read(frame)) break;
			flip(frame, frame, 1);
			imshow("原图", frame);
			moveWindow("原图", 30, 100);
			result = Cartoon(frame);

			imshow("卡通化", result);
			moveWindow("卡通化", 675, 100);
			char key = waitKey(1);
			if (key == 27) { //ESC退出
				break;
			}
			else if (key == 32) { //空格保存
				time_t t = time(0);
				strftime(ch, sizeof(ch), "%Y-%m-%d %H-%M-%S", localtime(&t));
				imwrite(path + ch + ".jpg", result);
			}
		}
	}
	return 0;
}