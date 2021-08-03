#include "MultiBandBlender.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

cv::Mat MultiBandBlender::Blend(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask, int levels)
{
    auto gaussPry1 = gaussianPyramid(img1, levels);
    auto gaussPry2 = gaussianPyramid(img2, levels);
    auto gaussPryMask = gaussianPyramid(mask, levels);

    auto lapPry1 = laplacianPyramid(gaussPry1);
    auto lapPry2 = laplacianPyramid(gaussPry2);

    auto blendPry = blendPyramid(lapPry1, lapPry2, gaussPryMask);

    return collapsePyramid(blendPry);
}

std::vector<cv::Mat> MultiBandBlender::gaussianPyramid(const cv::Mat& img, int levels)
{
    std::vector<cv::Mat> gaussianPyr;
    gaussianPyr.emplace_back(img);

    cv::Mat currentImg = img.clone();
    for (int i = 1; i < levels; ++i)
    {
        cv::pyrDown(currentImg, currentImg, cv::Size(currentImg.cols / 2, currentImg.rows / 2), cv::BORDER_REPLICATE);
        gaussianPyr.emplace_back(currentImg);
        
    }

    return gaussianPyr;
}

std::vector<cv::Mat> MultiBandBlender::laplacianPyramid(const std::vector<cv::Mat>& gaussianPyr)
{
    int levels = gaussianPyr.size();
    std::vector<cv::Mat> laplacianPyr;

    laplacianPyr.emplace_back(gaussianPyr[levels - 1]);
    
    for (int i = levels - 2; i >= 0; --i)
    {
        cv::Mat upsampleImg;
        cv::pyrUp(gaussianPyr[i + 1], upsampleImg, gaussianPyr[i].size());

        cv::Mat currentImg = gaussianPyr[i] - upsampleImg;
        laplacianPyr.emplace_back(currentImg);

        
    }

    return laplacianPyr;
}

std::vector<cv::Mat> MultiBandBlender::blendPyramid(const std::vector<cv::Mat>& pry1, const std::vector<cv::Mat>& pry2, const std::vector<cv::Mat>& pryMask)
{
    int levels = pry1.size();
    std::vector<cv::Mat> blendPry;

    for (int i = 0; i < levels; ++i)
    {
        cv::Mat blendImg = cv::Mat::zeros(pry1[i].size(), pry1[i].type());
        for (int row = 0; row < pry1[i].rows; ++row)
        {
            for (int col = 0; col < pry1[i].cols; ++col)
            {
                cv::Vec3f color1 = pry1[i].at<cv::Vec3f>(row, col);
                cv::Vec3f color2 = pry2[i].at<cv::Vec3f>(row, col);
                cv::Vec3f w = pryMask[levels - i - 1].at<cv::Vec3f>(row, col);

                cv::Vec3f newColor = color1 * (1 - w[2]) + color2 * w[2];
                blendImg.at<cv::Vec3f>(row, col) = newColor;
            }
        }
        //cv::Mat blendImg = pry1[i].mul(2 - pryMask[levels - i - 1]) + pry2[i].mul(pryMask[levels - i - 1]);
        blendPry.emplace_back(blendImg);
    }
 
    return blendPry;
}



cv::Mat MultiBandBlender::collapsePyramid(std::vector<cv::Mat> blendPyr)
{
    int levels = blendPyr.size();
    cv::Mat currentImg = blendPyr[0];

    for (int i = 1; i < levels; ++i)
    {
        cv::pyrUp(currentImg, currentImg, blendPyr[i].size());
        currentImg += blendPyr[i];
    }

    cv::Mat blendImg;
    cv::convertScaleAbs(currentImg, blendImg, 255);
    blendImg.convertTo(blendImg, CV_8UC3);
    return blendImg;
}
