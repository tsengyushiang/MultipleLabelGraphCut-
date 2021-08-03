#pragma once
#include <opencv2/core.hpp>
#include <vector>

class MultiBandBlender
{
public:
	MultiBandBlender() {}

	cv::Mat Blend(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask, int levels);
private:
	
	std::vector<cv::Mat> gaussianPyramid(const cv::Mat& img, int levels);
	std::vector<cv::Mat> laplacianPyramid(const std::vector<cv::Mat>& gaussianPyr);
	std::vector<cv::Mat> blendPyramid(const std::vector<cv::Mat>& pry1, const std::vector<cv::Mat>& pry2, const std::vector<cv::Mat>& pryMask);
	cv::Mat collapsePyramid(std::vector<cv::Mat> blendPyr);
};

