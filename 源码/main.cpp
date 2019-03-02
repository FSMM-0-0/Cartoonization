#include "stdafx.h"
#include "cartoon.h"
#include "highgui.h"
#include "cv.h"

void Init()
{
	cout << "����\"1 + picture_path\"ѡ��ͼƬ" << endl;
	cout << "����\"2\"������ͷ" << endl;
	cout << "����\"# + 0~4\"ʵ�ֲ�ͬ����ͼ��ͨ��" << endl;
	cout << "����ո񱣴�ͼƬ" << endl;
	cout << "���ESC�ر��˳�" << endl;
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
	if (t == 1) { //�򿪵���ͼ��
		cin >> path;
		img = imread(path);
		if (img.empty()) {
			cout << "ͼƬ����ʧ��" << endl;
		}
		else {
			cin >> c >> type;
			result = Cartoon(img, type);
			string str = "��ͨ��_" + to_string(type);
			imshow(str, result);
			char key = waitKey();

			if (key == 32) {
				time_t t = time(0);
				strftime(ch, sizeof(ch), "%Y-%m-%d %H-%M-%S", localtime(&t));
				imwrite(save_path + ch + ".jpg", result);
			}
		}
	}
	else if (t == 2) { //������ͷ
		namedWindow("�Զ�����", 0);
		createBar("�Զ�����");

		VideoCapture cap(0);
		Mat frame;
		while (cap.isOpened()) {
			if (!cap.read(frame)) break;
			flip(frame, frame, 1);
			imshow("ԭͼ", frame);
			moveWindow("ԭͼ", 30, 100);
			result = Cartoon(frame);

			imshow("��ͨ��", result);
			moveWindow("��ͨ��", 675, 100);
			char key = waitKey(1);
			if (key == 27) { //ESC�˳�
				break;
			}
			else if (key == 32) { //�ո񱣴�
				time_t t = time(0);
				strftime(ch, sizeof(ch), "%Y-%m-%d %H-%M-%S", localtime(&t));
				imwrite(path + ch + ".jpg", result);
			}
		}
	}
	return 0;
}