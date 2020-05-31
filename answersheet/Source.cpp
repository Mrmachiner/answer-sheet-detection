#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include<conio.h>
cv::Mat ResizeImage(cv::Mat img, int height = 800)
{
	//If you want 75 % along each axis, you should be able to use cv::resize to do:
	//cv::resize(inImg, outImg, cv::Size(), 0.75, 0.75);
	//cvtColor(dst, dst1, cv::COLOR_BGR2GRAY);
	float rat = height / (1.0 * img.size().height);
	int width = (int)(rat * img.size().width);
	cv::Mat dst = cv::Mat::ones(cv::Size((int)width, height), img.type());
	resize(img, dst, dst.size(), 0, 0, cv::INTER_AREA);
	return dst;
}
cv::Mat SharpenImage(cv::Mat img)
{
	cv::Mat dst;
	/*cv::Mat kernel = (cv::Mat_<char>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
	//other cv::Mat 0 -1 0 -1 5 -1 0 -1 0;
	filter2D(img, dst, img.depth(), kernel);*/
	GaussianBlur(img, dst, cv::Size(0, 0), 3);
	addWeighted(img, 1.5, dst, -0.5, 0, dst);
	return dst;
}
int FindPositionMin(float _array[], int length)
{
	int vt = 0;
	float min = _array[0];
	for (int i = 1; i < length; i++)
	{
		if (_array[i] < min)
		{
			min = _array[i];
			vt = i;
		}
	}
	return vt;
}
int FindPositionMax(float _array[], int length)
{
	int vt = 0;
	float max = _array[0];
	for (int i = 1; i < length; i++)
	{
		if (_array[i] > max)
		{
			max = _array[i];
			vt = i;
		}
	}
	return vt;
}
std::vector<cv::Point2f> SortCornerPoints(std::vector<cv::Point2f> points)
{
	std::vector<cv::Point2f> _pts(4);
	float* _sum = new float[points.size()];
	float* _diff = new float[points.size()];
	for (size_t i = 0; i < points.size(); i++)
	{
		_sum[i] = points[i].x + points[i].y;
	}
	for (size_t i = 0; i < points.size(); i++)
	{
		_diff[i] = points[i].y - points[i].x;
	}
	/*the top-left point will have the smallest sum, whereas
	the bottom - right point will have the largest sum
	now, compute the difference between the points, the
	top - right point will have the smallest difference,
	 whereas the bottom - left will have the largest difference*/
	int vt0 = FindPositionMin(_sum, points.size());
	int vt1 = FindPositionMin(_diff, points.size());
	int vt2 = FindPositionMax(_sum, points.size());
	int vt3 = FindPositionMax(_diff, points.size());
	_pts[0] = points[vt0];
	_pts[1] = points[vt1];
	_pts[2] = points[vt2];
	_pts[3] = points[vt3];
	return _pts;
}
bool Ycompare(const cv::Point2f p1, const cv::Point2f p2)
{
	return(p1.y + 5 < p2.y);
}
bool  Xcompare(const cv::Point2f p1, const cv::Point2f p2)
{
	return(p1.x + 5 < p2.x);
}
std::vector<cv::Point2f> SortPoints(std::vector<cv::Point2f> points, int axis = 0)
{
	if (axis == 0)
	{
		std::stable_sort(points.begin(), points.end(), Ycompare);
		std::stable_sort(points.begin(), points.end(), Xcompare);
	}
	else if (axis == 1)
	{
		std::stable_sort(points.begin(), points.end(), Xcompare);
		std::stable_sort(points.begin(), points.end(), Ycompare);
	}
	return points;
}
std::vector<cv::Point2f> TranformPoints(std::vector<cv::Point2f> points, cv::Point2f offset, float rat = 1)
{
	for (size_t i = 0; i < points.size(); i++)
	{
		points[i].x = (points[i].x * rat + offset.x);
		points[i].y = (points[i].y * rat + offset.y);
	}
	return points;
}
cv::Mat TranformImage(cv::Mat img, std::vector<cv::Point2f> points)
{
	float newHeight = std::max(norm(points[0] - points[3]), norm(points[1] - points[2]));
	float newWidth = std::max(norm(points[0] - points[1]), norm(points[2] - points[3]));
	std::vector<cv::Point2f> _dst;
	_dst.push_back(cv::Point2f(0, 0));
	_dst.push_back(cv::Point2f(newWidth - 1, 0));
	_dst.push_back(cv::Point2f(newWidth - 1, newHeight - 1));
	_dst.push_back(cv::Point2f(0, newHeight - 1));
	cv::Mat _M = getPerspectiveTransform(points, _dst);
	cv::Mat _Warp;
	warpPerspective(img, _Warp, _M, cv::Size((int)newWidth, (int)newHeight));
	return	_Warp;
}
cv::Mat DocummentScan(cv::Mat src, cv::Point2f offset)
{
	cv::Mat orig = src.clone();
	cv::Mat dst, dst1;
	std::vector< cv::Point2f> roi_corners;
	std::vector< cv::Point2f> dst_corners(4);
	int h = 800;
	float rat = src.size().height / (h * 1.0);
	src = ResizeImage(src, h);
	cvtColor(src, dst, cv::COLOR_BGR2RGB);
	cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
	bilateralFilter(dst, dst1, 9, 75, 75);
	adaptiveThreshold(dst1, dst1, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 115, 2);
	medianBlur(dst1, dst1, 11);
	copyMakeBorder(dst1, dst1, 5, 5, 5, 5, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	int height = src.size().height;
	int width = src.size().width;
	cv::Mat edges;
	Canny(dst1, edges, 200, 250);
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	int MAX_COUNTOUR_AREA = (width - 30) * (height - 30);
	double maxAreaFound = MAX_COUNTOUR_AREA * 0.1;
	roi_corners.push_back(cv::Point2f(5, 5));
	roi_corners.push_back(cv::Point2f((width - 5), 5));
	roi_corners.push_back(cv::Point2f((width - 5), (height - 5)));
	roi_corners.push_back(cv::Point2f(5, (height - 5)));
	for (size_t i = 0; i < contours.size(); i++)
	{
		double perimeter = arcLength(contours[i], true);
		std::vector<cv::Point2f> approx;
		approxPolyDP(contours[i], approx, 0.03 * perimeter, true);
		bool isConvex = isContourConvex(approx);
		double area = contourArea(approx);
		bool ok = (approx.size() == 4 && isContourConvex(approx) && maxAreaFound < contourArea(approx) && contourArea(approx) < MAX_COUNTOUR_AREA);
		if (ok)
		{
			maxAreaFound = contourArea(approx);
			roi_corners = approx;
		}
	}
	roi_corners = SortCornerPoints(roi_corners);
	dst_corners = TranformPoints(roi_corners, cv::Point2f(), rat);
	cv::Mat i = TranformImage(orig, dst_corners);
	std::vector<cv::Point2f> newDst_corners;
	newDst_corners.push_back(cv::Point2f(offset.y, offset.x));
	newDst_corners.push_back(cv::Point2f(i.size().width - offset.y, offset.x));
	newDst_corners.push_back(cv::Point2f(i.size().width - offset.y, i.size().height - offset.x));
	newDst_corners.push_back(cv::Point2f(offset.y, i.size().height - offset.x));
	cv::Mat newImage = TranformImage(i, newDst_corners);
	return newImage;
}
std::vector<std::vector<cv::Point2f>> FindAnchors(cv::Mat img, double Area = INT_MAX, double deltaArea = INT_MAX)
{
	std::vector<std::vector<cv::Point2f>> anchors;
	cv::Mat src_HSV;
	cvtColor(img, src_HSV, cv::COLOR_BGR2HSV);
	blur(src_HSV, src_HSV, cv::Size(5, 5));
	cv::Mat edges;
	inRange(src_HSV, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 50), edges);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	/// Find contours
	findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	//imshow("findContours", edges);
	for (int i = 0; i < contours.size(); i++)
	{
		double perimeter = arcLength(contours[i], true);
		std::vector<cv::Point2f> approx;
		approxPolyDP(contours[i], approx, 0.03 * perimeter, true);
		double area = contourArea(contours[i]);
		bool ok;
		if (Area == INT_MAX || deltaArea == INT_MAX)
		{
			ok = (approx.size() == 4 && isContourConvex(approx));
		}
		else
		{
			ok = (approx.size() == 4 && (area > Area - deltaArea) && (area < Area + deltaArea) && isContourConvex(approx));
		}
		if (ok)
		{
			anchors.push_back(approx);
		}
	}
	return anchors;
}
std::vector<std::vector<cv::Point2f>> FindRectangles(cv::Mat img, double fromThreshold, double toThreshold, double ratioArea = 0)
{
	std::vector<std::vector<cv::Point2f>> rectangles;
	cv::Mat src_gray;
	GaussianBlur(img, img, cv::Size(3, 3), 0);
	cvtColor(img, src_gray, cv::COLOR_BGR2RGB);
	cvtColor(src_gray, src_gray, cv::COLOR_BGR2GRAY);
	cv::Mat dst1;
	bilateralFilter(src_gray, dst1, 9, 75, 75);
	adaptiveThreshold(dst1, dst1, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 115, 2);
	medianBlur(dst1, dst1, 11);
	cv::Mat edges;
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	Canny(src_gray, edges, fromThreshold, toThreshold, 3);
	double Max_Area_Find = (img.size().width - 10) * (img.size().height - 10);
	/// Find contours
	findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	//imshow("findcontour", edges);
	std::vector<cv::Point2f> approx;
	//imshow("findContours+Ed", edges);
	for (int i = 0; i < contours.size(); i++)
	{
		double perimeter = arcLength(contours[i], true);
		approxPolyDP(contours[i], approx, 0.03 * perimeter, true);
		double area = contourArea(contours[i]);
		bool ok = false;
		if (ratioArea == 0)
		{
			ok = approx.size() == 4 && isContourConvex(approx);
		}
		else
		{
			ok = approx.size() == 4 && isContourConvex(approx) && area >= ratioArea * Max_Area_Find; //ratioArea * Max_Area_Find
		}
		if (ok)
		{
			rectangles.push_back(approx);
		}
	}
	return rectangles;
}
std::vector<std::vector<cv::Point2f>>  TranformContours(std::vector<std::vector<cv::Point2f>> contours)
{
	// sort 4 conner contour
	std::vector<std::vector<cv::Point2f>> temp = contours;
	for (size_t i = 0; i < contours.size(); i++)
	{
		temp[i] = SortCornerPoints(contours[i]);
	}
	return temp;
}
cv::Point2f PointIntersecsion(std::vector<cv::Point2f> points)
{
	cv::Moments M;
	M = moments(points);
	int x = int(M.m10 / M.m00);
	int y = int(M.m01 / M.m00);
	//giao nhau
	return cv::Point(x, y);
}
std::vector<cv::Point2f> ListPointIntersection(std::vector<std::vector<cv::Point2f>> contours)
{
	std::vector<cv::Point2f> dst;
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Point2f p = PointIntersecsion(contours[i]);
		dst.push_back(p);
	}
	return dst;
}
cv::Point2f FindPointInRegion(std::vector<cv::Point2f> points, cv::Point2f condition1, cv::Point2f condition2, int axis)
{
	// axis =0,1;  1 theo chi?u x,0 theo chi?u y
	cv::Point2f point;
	for (size_t i = 0; i < points.size(); i++)
	{
		if (axis == 1)
		{
			// tìm point trong khoang delta y
			float average = (condition1.y + condition2.y) / 2.0;
			float delta = std::fabs(condition1.y - condition2.y) + 5;
			if (points[i].y == average)
			{
				point = points[i];
				break;
			}
			else
			{
				bool ok = points[i].y > (average - delta) && points[i].y < (average + delta);
				if (ok)
				{
					point = points[i];
					break;
				}
			}
		}
		else
		{
			//tìm point trong kho?ng delta x
			float average = (condition1.x + condition2.x) / 2.0;
			float delta = std::fabs(condition1.x - condition2.x) + 5;
			if (points[i].x == average)
			{
				point = points[i];
				break;
			}
			else
			{
				bool ok = points[i].x > (average - delta) && points[i].x < (average + delta);
				if (ok)
				{
					point = points[i];
					break;
				}
			}
		}
	}
	return point;
}
double Distance(cv::Point2f p1, cv::Point2f p2)
{
	double d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
	return std::sqrtf(d);
}
std::vector<cv::Point2f> ClusterPoints(std::vector<cv::Point2f>  points, double distance = 3)
{
	//Gom Tam
	std::vector<cv::Point2f> dst;
	while (points.size() != 0)
	{
		int _count = 1;
		cv::Point2f p = points[0];
		int sumX = p.x;
		int sumY = p.y;
		points.erase(points.begin());
		for (size_t i = 0; i < points.size(); i++)
		{
			if (p == points[i])
			{
				points.erase(points.begin() + i);
				i--;
			}
			else if (Distance(p, points[i]) < distance)
			{
				sumX += points[i].x;
				sumY += points[i].y;
				_count++;
				points.erase(points.begin() + i);
				i--;
			}
		}
		dst.push_back(cv::Point2f(int(sumX / _count * 1.0), int(sumY / _count * 1.0)));
	}
	return dst;
}
std::vector<cv::Point2f> SortAnchors(std::vector<cv::Point2f> src)
{
	std::vector<cv::Point2f> dst;
	std::vector<cv::Point2f> corners = SortCornerPoints(src);
	for (size_t i = 0; i < corners.size(); i++)
	{
		src.erase(std::remove(src.begin(), src.end(), corners[i]));
	}
	//corner 1
	dst.push_back(corners[0]);
	cv::Point2f point = FindPointInRegion(src, corners[0], corners[1], 1);
	dst.push_back(point);
	src.erase(std::remove(src.begin(), src.end(), point));
	//corner 3
	dst.push_back(corners[1]);
	point = FindPointInRegion(src, corners[1], corners[2], 0);
	dst.push_back(point);
	src.erase(std::remove(src.begin(), src.end(), point));
	//corner 5
	dst.push_back(corners[2]);
	dst.push_back(corners[3]);
	point = FindPointInRegion(src, corners[0], corners[3], 0);
	dst.push_back(point);
	src.erase(std::remove(src.begin(), src.end(), point));
	//corner 7
	dst.push_back(src[0]);
	return dst;
}
std::vector<std::vector<cv::Point2f>> SortRectangles(std::vector<std::vector<cv::Point2f>> rects, int axis = 0)
{
	std::vector<std::vector<cv::Point2f>> rectangles;
	std::vector<cv::Point2f> centerPoints;
	for (size_t i = 0; i < rects.size(); i++)
	{
		cv::Point2f p = PointIntersecsion(rects[i]);
		centerPoints.push_back(p);
	}
	std::vector<cv::Point2f> _copy = centerPoints;
	_copy = ClusterPoints(_copy);
	_copy = SortPoints(_copy, axis);
	for (size_t i = 0; i < _copy.size(); i++)
	{
		for (size_t j = 0; i < centerPoints.size(); j++)
		{
			if (_copy[i] == centerPoints[j])
			{
				rectangles.push_back(rects[j]);
				centerPoints.erase(centerPoints.begin() + j);
				break;
			}
			else
			{
				if (Distance(_copy[i], centerPoints[j]) < 3)
				{
					rectangles.push_back(rects[j]);
					centerPoints.erase(centerPoints.begin() + j);
					break;
				}
			}
		}
	}
	return rectangles;
}
cv::Mat SubRectangleImage(cv::Mat src, std::vector<cv::Point2f> roi)
{
	cv::Rect rect = boundingRect(roi);
	cv::Mat subimage = src(rect).clone();
	return subimage;
}
cv::Mat SubCircleImage(cv::Mat src, cv::Point2f center, double radius)
{
	cv::Mat subImage = src(cv::Rect(center.x - radius, // ROI x-offset, left coordinate
		center.y - radius, // ROI y-offset, top coordinate 
		2 * radius,          // ROI width
		2 * radius)).clone();
	return subImage;
}
std::vector<cv::Mat> SeriImage(cv::Mat src, std::vector<std::vector<cv::Point2f>> rects)
{
	//ofstream toaDo("ToaDoVungDA.csv");
	std::vector<cv::Mat> serious;
	for (size_t i = 0; i < rects.size(); i++)
	{
		std::vector<cv::Point2f> cornners = SortCornerPoints(rects[i]);
		//if (rects.size() == 4) {
		//	for (int j = 0; j < cornners.size(); j++) {
		//		toaDo << cornners[j].x << "," << cornners[j].y << endl;
		//	}
		//}
		cv::Mat M = TranformImage(src, cornners);
		serious.push_back(M);
	}
	return serious;
}
std::vector<std::vector<cv::Point2f>> FindCircle(cv::Mat img, double adaptiveThresHold, double Area = INT_MAX, double deltaArea = INT_MAX)
{
	std::vector<std::vector<cv::Point2f>> circle;
	cv::Size size(3, 3);
	cv::GaussianBlur(img, img, size, 0);
	cv::Mat dst;
	bilateralFilter(img, dst, 9, 75, 75);
	img = dst;
	cvtColor(img, img, cv::COLOR_BGR2GRAY);
	adaptiveThreshold(img, img, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 75, 10);
	cv::bitwise_not(img, img);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	/// Find contours
	findContours(img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	for (int i = 0; i < contours.size(); i++)
	{
		double perimeter = arcLength(contours[i], true);
		std::vector<cv::Point2f> approx;
		approxPolyDP(contours[i], approx, 0.03 * perimeter, true);
		double area = contourArea(contours[i]);
		bool ok;
		if (area == INT_MAX || deltaArea == INT_MAX)
		{
			ok = approx.size() >= 6 && isContourConvex(approx);
		}
		else
		{
			ok = approx.size() >= 6 && (area > Area - deltaArea) && (area < Area + deltaArea) && isContourConvex(approx);
		}
		if (ok)
		{
			circle.push_back(approx);
		}
	}
	return circle;
}
double FindRatioWhiteRegion(cv::Mat src, int threadhold = 185)
{
	cv::Mat src_gray;
	cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
	blur(src_gray, src_gray, cv::Size(3, 3));
	threshold(src_gray, src_gray, threadhold, 255, cv::THRESH_BINARY);
	countNonZero(src_gray);
	int white = countNonZero(src_gray);
	double ratio = white / (src_gray.size().width * 1.0 * src_gray.size().height * 1.0);
	return ratio;
}
std::vector<int> FindChoice(cv::Mat src, std::vector<cv::Point2f> points, int radious = 10)
{
	std::vector<int> dst;
	for (size_t i = 0; i < points.size(); i++)
	{
		cv::Mat img = SubCircleImage(src, points[i], radious);
		//imshow("FindChoice"+i, img);
		double _ratio = FindRatioWhiteRegion(img);
		if (_ratio <= 0.35) //       <=
		{
			dst.push_back(1);
		}
		else
		{
			dst.push_back(0);
		}
	}
	return dst;
}
std::vector<int> AnswerChoice(cv::Mat src, double adaptiveThresHold = 75, int axis = 0)
{
	//cv::Mat drawing = cv::Mat::zeros(src.size(), CV_8UC3);
	std::vector<std::vector<cv::Point2f>> circles = FindCircle(src, adaptiveThresHold);
	std::vector<cv::Point2f> CenterPoints = ListPointIntersection(circles);
	CenterPoints = ClusterPoints(CenterPoints, 3);
	CenterPoints = SortPoints(CenterPoints, axis);
	/*for (int i = 0; i < CenterPoints.size(); i++)
	{
		imshow("drawing", drawing);
		circle(drawing, CenterPoints[i], 2, cv::Scalar(0, 255, 0), 2);
		waitKey(300);
	}*/
	std::vector<int> choices = FindChoice(src, CenterPoints);
	return choices;
}
std::vector<std::string> CharacterResults(std::vector<int> answerChoice, int optionChoices = 4)
{
	std::vector<char> characters;
	int ch = 65;
	for (int i = 0; i < optionChoices; i++)
	{
		char x(ch);
		characters.push_back(x);
		ch++;
	}
	std::vector<std::string> results;
	int _count = 0;
	std::string str = "";
	for (int i = 0; i < answerChoice.size(); i++)
	{
		_count++;
		if (answerChoice[i] == 1)
		{
			str.append(1, characters[_count - 1]);
			str += " ";
		}
		if (_count >= optionChoices)
		{
			results.push_back(str);
			str = "";
			_count = 0;
		}
	}
	return results;
}
std::string NumberResults(std::vector<int> answerChoice, int optionChoices = 10)
{
	int _count = 0;
	std::string str = "";
	for (int i = 0; i < answerChoice.size(); i++)
	{
		_count++;
		if (answerChoice[i] == 1)
		{
			str += std::to_string(_count - 1);
		}
		if (_count >= optionChoices)
		{
			_count = 0;
		}
	}
	return str;
}
std::vector<std::string> IdentityHeader(std::vector<cv::Mat> Mats, int optionChoice = 10)
{
	std::vector<std::string> indentity;
	for (size_t i = 0; i < Mats.size(); i++)
	{
		std::vector<int> ac = AnswerChoice(ResizeImage(Mats[i]), optionChoice);
		std::string str = NumberResults(ac);
		indentity.push_back(str);
	}
	return indentity;
}
std::vector<std::vector<std::string>> AllAnswers(std::vector<cv::Mat> Mats, int optionChoice = 4)
{
	std::vector<std::vector<std::string>> allAnswers;
	for (size_t i = 0; i < Mats.size(); i++)
	{
		std::vector<int> ac = AnswerChoice(ResizeImage(Mats[i]), 75, 1);
		std::vector<std::string> str = CharacterResults(ac, optionChoice);
		allAnswers.push_back(str);
	}
	return allAnswers;
}
void main(int argc, char** argv)
{
	cv::Mat src = cv::imread("C:\\Users\\T450s\\source\\repos\\answersheet\\image_demo\\2.jpeg"); //err 4.jpeg 5.jpg 7.jpg 8.jpg
	cv::namedWindow("src", cv::WINDOW_GUI_NORMAL);
	cv::Mat _copy = ResizeImage(src);
	imshow("src", _copy);
	//neo
	std::vector<std::vector<cv::Point2f>> contours = FindAnchors(_copy);
	std::vector<cv::Point2f> lstPointIntersection = ListPointIntersection(contours);
	//c?m
	lstPointIntersection = ClusterPoints(lstPointIntersection, 3);
	std::vector<cv::Point2f> anchors;
	anchors = SortAnchors(lstPointIntersection);
	anchors = TranformPoints(anchors, cv::Point2f(), src.size().height / 800.0);
	std::vector<cv::Point2f> indentityRegion = { anchors[1],anchors[3] };
	std::vector<cv::Point2f> answerRegion = { anchors[3],anchors[5] };
	cv::Mat subIdImage = SubRectangleImage(src, indentityRegion);
	subIdImage = ResizeImage(subIdImage);
	cv::Mat subAnswerImage = SubRectangleImage(src, answerRegion);
	//imwrite("C:\\Users\\MinhHoang\\Desktop\\Source_Img\\Asware.jpg", subAnswerImage);
	subAnswerImage = ResizeImage(subAnswerImage);
	//imwrite("C:\\Users\\MinhHoang\\Desktop\\Source_Img\\aw.jpg", subAnswerImage);
	cv::namedWindow("g", cv::WINDOW_GUI_NORMAL);
	cv::namedWindow("gg", cv::WINDOW_GUI_NORMAL);
	std::vector<std::vector<cv::Point2f>> IdRects = FindRectangles(subIdImage, 200, 250, 0.05);
	IdRects = SortRectangles(IdRects, 1);
	std::vector<std::vector<cv::Point2f>> answerRects = FindRectangles(subAnswerImage, 200, 250, 0.05);
	answerRects = SortRectangles(answerRects, 1);
	//ofstream toaDo("ToaDoVungDA.csv");
	//for (int i = 0; i < answerRects.size(); i++) 
	//{
	//	toaDo << answerRects[i][0].x << "," << answerRects[i][0].y << endl;
	//	toaDo << answerRects[i][1].x << "," << answerRects[i][1].y << endl;
	//	toaDo << answerRects[i][2].x << "," << answerRects[i][2].y << endl;
	//	toaDo << answerRects[i][3].x << "," << answerRects[i][3].y << endl;
	//}
	imshow("g", subIdImage);
	imshow("gg", subAnswerImage);

	std::vector<cv::Mat> IdMats = SeriImage(subIdImage, IdRects);
	std::vector<cv::Mat> AnswerMats = SeriImage(subAnswerImage, answerRects);
	for (int i = 0; i < AnswerMats.size(); i++) {
		cv::imshow("88." + std::to_string(i) + ".jpg", AnswerMats[i]);
		//cv::imwrite("C:\\Users\\MinhHoang\\Desktop\\Source_Img\\88." + std::to_string(i) + ".jpg", AnswerMats[i]);
	}
	//for (int i = 0; i < Answercv::Mats.size(); i++)
	//{
	//	imshow("abcd" + i, Answercv::Mats[i]);
	//	imwrite(sourceImag.append("ID" + i + name), Answercv::Mats[i]);
	//}
	//imwrite("C:\\Users\\MinhHoang\\Desktop\\Source_Img\\x3.jpg", Answercv::Mats[3]);
	std::vector<std::string> id = IdentityHeader(IdMats);
	std::vector<std::vector<std::string>> allAnswers = AllAnswers(AnswerMats);
	std::cout << "SBD :" << id[0] << std::endl;
	std::cout << "Ma De :" << id[1] << std::endl;
	int k = 1;
	for (int i = 0; i < allAnswers.size(); i++)
	{
		for (int j = 0; j < allAnswers[i].size(); j++)
		{
			std::cout << k << "." << allAnswers[i][j] << "     ";
			k++;
		}
	}
	cv::waitKey();
}
