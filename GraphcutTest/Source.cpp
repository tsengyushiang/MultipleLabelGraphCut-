#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <GCoptimization.h>
#include <vector>
#include <limits>
#include "MultiBandBlender.h"
#include <opencv2/stitching.hpp>
#include <unordered_map>

cv::Mat CreateMask(const cv::Mat& img, cv::int32_t value)
{
	if (img.empty())
	{
		return cv::Mat();
	}

	cv::Mat mask(img.size(), CV_32SC1);
	for (int row = 0; row < img.rows; ++row)
	{
		for (int col = 0; col < img.cols; ++col)
		{
			auto color = img.at<cv::Vec4b>(row, col);
			if (color[3] > 0)
			{
				mask.at<cv::int32_t>(row, col) = value;
			}
		}
	}

	return mask;
}

bool IsOverlapping(cv::int32_t value, cv::int32_t* imgIDs, int length)
{
	for (int i = 0; i < length; ++i)
	{
		if (value <= 0 || value == imgIDs[i])
		{
			return false;
		}
	}

	return true;
}

bool AtImgBound(const cv::Mat& overlapMask, int x, int y, cv::int32_t value)
{
	for (int offsetY = -1; offsetY < 2; ++offsetY)
	{
		for (int offsetX = -1; offsetX < 2; ++offsetX)
		{
			int newX = x + offsetX;
			int newY = y + offsetY;
			if (newX >= 0 && newX < overlapMask.cols &&
				newY >= 0 && newY < overlapMask.rows)
			{
				if (overlapMask.at<cv::int32_t>(newY, newX) == value)
				{
					return true;
				}
			}
		}
	}
	return false;
}

bool InRange(const cv::Mat& img, int x, int y)
{
	return (x >= 0 && x < img.cols) && (y >= 0 && y < img.rows);
}

struct EdgeCostFunctor : GCoptimization::SmoothCostFunctor
{
	cv::Mat* img;
	cv::Rect* imgRect;
	cv::int32_t* imgIDs;
	std::unordered_map<int, cv::Point2i> lot;
	cv::Rect overlapRect;
	cv::Mat overlapMask;

	EdgeCostFunctor()
	{
	}

	virtual GCoptimization::EnergyTermType compute(GCoptimization::SiteID s1, GCoptimization::SiteID s2, GCoptimization::LabelID l1, GCoptimization::LabelID l2)
	{
		if (l1 == l2)
		{
			return 0;
		}

		cv::Point2i p1 = lot[s1];
		cv::Point2i p2 = lot[s2];
		//std::cout << p1 << " " << p2 << std::endl;

		if (InRange(img[l1], overlapRect.x + p1.x - imgRect[l1].x, overlapRect.y + p1.y - imgRect[l1].y) && InRange(img[l1], overlapRect.x + p2.x - imgRect[l1].x, overlapRect.y + p2.y - imgRect[l1].y) &&
			InRange(img[l2], overlapRect.x + p1.x - imgRect[l2].x, overlapRect.y + p1.y - imgRect[l2].y) && InRange(img[l2], overlapRect.x + p2.x - imgRect[l2].x, overlapRect.y + p2.y - imgRect[l2].y)
			)
		{
			if ((overlapMask.at<cv::int32_t>(p1.y + overlapRect.y, p1.x + overlapRect.x) & imgIDs[l1]) > 0 && (overlapMask.at<cv::int32_t>(p2.y + overlapRect.y, p2.x + overlapRect.x) & imgIDs[l1]) > 0 &&
				(overlapMask.at<cv::int32_t>(p1.y + overlapRect.y, p1.x + overlapRect.x) & imgIDs[l2]) > 0 && (overlapMask.at<cv::int32_t>(p2.y + overlapRect.y, p2.x + overlapRect.x) & imgIDs[l2]) > 0)
			{

				cv::Vec4b colorS1(0, 0, 0);
				cv::Vec4b colorT1(0, 0, 0);
				cv::Vec4i gradST1(0, 0, 0);

				cv::Vec4b colorS2(0, 0, 0);
				cv::Vec4b colorT2(0, 0, 0);
				cv::Vec4i gradST2(0, 0, 0);
				cv::Vec4i gradS(0, 0, 0);
				cv::Vec4i gradT(0, 0, 0);
				bool p1InRange = false, p2InRange = false;
				if (InRange(img[l1], overlapRect.x + p1.x - imgRect[l1].x, overlapRect.y + p1.y - imgRect[l1].y) && InRange(img[l1], overlapRect.x + p2.x - imgRect[l1].x, overlapRect.y + p2.y - imgRect[l1].y))
				{
					colorS1 = img[l1].at<cv::Vec4b>(overlapRect.y + p1.y - imgRect[l1].y, overlapRect.x + p1.x - imgRect[l1].x);
					colorS2 = img[l1].at<cv::Vec4b>(overlapRect.y + p2.y - imgRect[l1].y, overlapRect.x + p2.x - imgRect[l1].x);
					gradS = (cv::Vec4i)colorS2 - (cv::Vec4i)colorS1;
					p1InRange = true;
				}

				if (InRange(img[l2], overlapRect.x + p1.x - imgRect[l2].x, overlapRect.y + p1.y - imgRect[l2].y) && InRange(img[l2], overlapRect.x + p2.x - imgRect[l2].x, overlapRect.y + p2.y - imgRect[l2].y))
				{
					colorT1 = img[l2].at<cv::Vec4b>(overlapRect.y + p1.y - imgRect[l2].y, overlapRect.x + p1.x - imgRect[l2].x);
					colorT2 = img[l2].at<cv::Vec4b>(overlapRect.y + p2.y - imgRect[l2].y, overlapRect.x + p2.x - imgRect[l2].x);
					gradT = (cv::Vec4i)colorT2 - (cv::Vec4i)colorT1;
					p2InRange = true;
				}

				gradST1 = (cv::Vec4i)colorT1 - (cv::Vec4i)colorS1;
				gradST2 = (cv::Vec4i)colorT2 - (cv::Vec4i)colorS2;


				double l2Norm = cv::sqrt(gradST1[0] * gradST1[0] + gradST1[1] * gradST1[1] + gradST1[2] * gradST1[2]) + cv::sqrt(gradST2[0] * gradST2[0] + gradST2[1] * gradST2[1] + gradST2[2] * gradST2[2]);
				double l2Weight = cv::sqrt(gradS[0] * gradS[0] + gradS[1] * gradS[1] + gradS[2] * gradS[2]) + cv::sqrt(gradT[0] * gradT[0] + gradT[1] * gradT[1] + gradT[2] * gradT[2]);
				//return l2Norm;
				int gb = cv::abs(gradST1[0]) + cv::abs(gradST2[0]);
				int gg = cv::abs(gradST1[1]) + cv::abs(gradST2[1]);
				int gr = cv::abs(gradST1[2]) + cv::abs(gradST2[2]);
				double normalizeWeight = (cv::abs(gradS[0]) + cv::abs(gradS[1]) + cv::abs(gradS[2])) / 3 + (cv::abs(gradT[0]) + cv::abs(gradT[1]) + cv::abs(gradT[2])) / 3;

				return (gb + gg + gr) / 3.0 / cv::sqrt(normalizeWeight + 1);
			}
			else
			{
				return 500;
			}


		}
		else
		{
			return 500;
		}
	}
};

int main(int argc, char* argv[])
{
	const int numImg = 4;

	cv::Vec3b colors[4] = { cv::Vec3b(255, 0, 0), cv::Vec3b(0, 255, 0) ,cv::Vec3b(0, 0, 255) ,cv::Vec3b(255, 255, 0) };
	cv::Mat img[numImg];
	img[0] = cv::imread("Imgs/R_IMG000.png", cv::IMREAD_UNCHANGED);
	img[1] = cv::imread("Imgs/R_IMG001.png", cv::IMREAD_UNCHANGED);
	img[2] = cv::imread("Imgs/R_IMG002.png", cv::IMREAD_UNCHANGED);
	img[3] = cv::imread("Imgs/R_IMG003.png", cv::IMREAD_UNCHANGED);
	cv::Point2i offset[numImg] = { cv::Point2i(0, 350), cv::Point2i(320, 374), cv::Point2i(4, 71), cv::Point2i(490, 0) };
	cv::Rect imgRect[numImg];
	int outputWidth = 0, outputHeight = 0;
	for (int i = 0; i < numImg; ++i)
	{
		imgRect[i] = cv::Rect(offset[i].x, offset[i].y, img[i].cols, img[i].rows);
		outputWidth = cv::max(outputWidth, imgRect[i].x + imgRect[i].width);
		outputHeight = cv::max(outputHeight, imgRect[i].y + imgRect[i].height);
	}

	cv::int32_t imgIDs[numImg] = { 1, 2, 4, 8 };

	// overlapping info
	cv::Mat overlapMask(outputHeight, outputWidth, CV_32SC1);
	for (int i = 0; i < numImg; ++i)
	{
		cv::Mat mask = CreateMask(img[i], imgIDs[i]);
		cv::add(overlapMask(imgRect[i]), mask, overlapMask(imgRect[i]));
	}

	int numOverlappingPixel = 0;
	cv::Rect overlapRect(outputWidth, outputHeight, 0, 0);
	for (int row = 0; row < outputHeight; ++row)
	{
		for (int col = 0; col < outputWidth; ++col)
		{
			if (IsOverlapping(overlapMask.at<cv::int32_t>(row, col), imgIDs, numImg))
			{
				overlapRect.x = cv::min(overlapRect.x, col);
				overlapRect.y = cv::min(overlapRect.y, row);
				overlapRect.width = cv::max(overlapRect.width, col);
				overlapRect.height = cv::max(overlapRect.height, row);
				++numOverlappingPixel;
			}
		}
	}
	overlapRect.width = overlapRect.width - overlapRect.x + 1;
	overlapRect.height = overlapRect.height - overlapRect.y + 1;

	// set graph
	cv::Mat visualizeData(overlapRect.height, overlapRect.width, CV_8UC3, cv::Scalar::all(0));

	int nodeIdx = -1;
	GCoptimizationGeneralGraph g(overlapRect.height * overlapRect.width, numImg);
	g.setLabelOrder(true);


	cv::Mat lot(overlapRect.height, overlapRect.width, CV_32SC1, cv::Scalar::all(0));
	std::unordered_map<int, cv::Point2i> pLot;
	std::vector<GCoptimization::SparseDataCost> dataCosts[numImg];
	for (int row = 0; row < overlapRect.height; ++row)
	{
		for (int col = 0; col < overlapRect.width; ++col)
		{
			int globalRow = overlapRect.y + row;
			int globalCol = overlapRect.x + col;
			//if (IsOverlapping(overlapMask.at<cv::int32_t>(globalRow, globalCol), imgIDs, numImg))
			{
				// add node
				++nodeIdx;
				lot.at<cv::int32_t>(row, col) = nodeIdx + 1;
				pLot[nodeIdx] = cv::Point2i(col, row);

				// set data term
				for (int i = 0; i < numImg; ++i)
				{
					if (AtImgBound(overlapMask, globalCol, globalRow, imgIDs[i]))
					{
						// connect to t-link
						dataCosts[i].push_back({ nodeIdx, 0.0 });
						visualizeData.at<cv::Vec3b>(row, col) = colors[i];
					}
					else
					{
						// disconnect t-link
						dataCosts[i].push_back({ nodeIdx, 500.0 });
					}
				}

				// set edge connection
				// x direction
				if (col > 0 && IsOverlapping(overlapMask.at<cv::int32_t>(globalRow, globalCol - 1), imgIDs, numImg))
				{
					int nID = lot.at<cv::int32_t>(row, col - 1) - 1;
					g.setNeighbors(nodeIdx, nID, 1);
				}

				// y direction
				if (row > 0 && IsOverlapping(overlapMask.at<cv::int32_t>(globalRow - 1, globalCol), imgIDs, numImg))
				{
					int nID = lot.at<cv::int32_t>(row - 1, col) - 1;
					g.setNeighbors(nodeIdx, nID, 1);
				}
			}
		}
	}

	for (int i = 0; i < numImg; ++i)
	{
		g.setDataCost(i, dataCosts[i].data(), dataCosts[i].size());
	}

	// cost function
	EdgeCostFunctor functor;
	functor.img = img;
	functor.imgIDs = imgIDs;
	functor.imgRect = imgRect;
	functor.lot = pLot;
	functor.overlapRect = overlapRect;
	functor.overlapMask = overlapMask.clone();
	g.setSmoothCostFunctor(&functor);
	g.setVerbosity(2);

	try
	{
		std::cout << "E1 " << g.compute_energy() << std::endl;
		g.swap(2); // -1 for minimum cost
		std::cout << "E2 " << g.compute_energy() << std::endl;
	}
	catch (GCException e)
	{
		e.Report();
	}


	cv::Mat seamMask(overlapRect.height, overlapRect.width, CV_8UC1, cv::Scalar::all(0));
	cv::Mat output(overlapRect.height, overlapRect.width, CV_8UC4, cv::Scalar::all(0));
	for (int row = 0; row < overlapRect.height; ++row)
	{
		for (int col = 0; col < overlapRect.width; ++col)
		{
			int nodeIdx = lot.at<cv::int32_t>(row, col);
			if (nodeIdx > 0)
			{
				int imgLabel = g.whatLabel(nodeIdx - 1);
				seamMask.at<cv::uint8_t>(row, col) = imgLabel + 1;
				visualizeData.at<cv::Vec3b>(row, col) = colors[imgLabel];
				output.at<cv::Vec4b>(row, col) = img[imgLabel].at<cv::Vec4b>(row + overlapRect.y - imgRect[imgLabel].y, col + overlapRect.x - imgRect[imgLabel].x);
			}
		}
	}

	cv::imwrite("output.png", output);
	cv::imwrite("visualizeData.png", visualizeData);

	return 0;
}