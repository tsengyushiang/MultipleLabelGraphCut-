#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <GCoptimization.h>
#include <vector>
#include <limits>
#include <Eigen/Dense>
#include <unsupported/Eigen/LevenbergMarquardt>
#include "MultiBandBlender.h"
#include <opencv2/stitching.hpp>

cv::Mat CreateMask(const cv::Mat& img, cv::uint8_t value)
{
	if (img.empty())
	{
		return cv::Mat();
	}

	cv::Mat mask(img.size(), CV_8UC1);
	for (int row = 0; row < img.rows; ++row)
	{
		for (int col = 0; col < img.cols; ++col)
		{
			auto color = img.at<cv::Vec3b>(row, col);
			if (color[0] > 0 || color[1] > 0 || color[2] > 0)
			{
				mask.at<cv::uint8_t>(row, col) = value;
			}
		}
	}

	return mask;
}

//// Generic functor
//template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
//struct Functor
//{
//	typedef _Scalar Scalar;
//	enum {
//		InputsAtCompileTime = NX,
//		ValuesAtCompileTime = NY
//	};
//	typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
//	typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
//	typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;
//
//	const int m_inputs, m_values;
//
//	Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
//	Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}
//
//	int inputs() const { return m_inputs; }
//	int values() const { return m_values; }
//
//	// you should define that in the subclass :
//  //  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
//};



struct LMFunctor : Eigen::DenseFunctor<double>
{
	const double omegaN = 3.0 * 3.0;
	const double omegaG = 0.1;
	int m; // num of equation
	int n; // num of parameter
	int count; // num of pixel in overlapping region
	cv::Vec3d avgItensity1, avgItensity2;
	LMFunctor(const cv::Mat& img1, const cv::Mat& img2, cv::Rect overlapRect, cv::Rect rect1, cv::Rect rect2)
	{
		m = 2;
		n = 2;

		avgItensity1 = cv::Vec3d{ 0.0, 0.0, 0.0 };
		avgItensity2 = cv::Vec3d{ 0.0, 0.0, 0.0 };
		count = 0;
		for (int row = 0; row < overlapRect.height; ++row)
		{
			for (int col = 0; col < overlapRect.width; ++col)
			{
				int newX1 = overlapRect.x + col - rect1.x, newY1 = overlapRect.y + row - rect1.y;
				int newX2 = overlapRect.x + col - rect2.x, newY2 = overlapRect.y + row - rect2.y;

				if (InRange(img1, newX1, newY1) &&
					InRange(img2, newX2, newY2))
				{
					cv::Vec3b color1 = img1.at<cv::Vec3b>(newY1, newX1);
					cv::Vec3b color2 = img2.at<cv::Vec3b>(newY2, newX2);
					if (HasColor(color1) && HasColor(color2))
					{
						avgItensity1 += cv::Vec3b(color1[2], color1[2], color1[2]);
						avgItensity2 += cv::Vec3b(color2[2], color2[2], color2[2]);
						++count;
					}
				}
			}
		}

		avgItensity1 /= count;
		avgItensity2 /= count;

		std::cout << avgItensity1 << std::endl;
		std::cout << avgItensity2 << std::endl;
	}

	bool InRange(const cv::Mat& img, int x, int y)
	{
		return (x >= 0 && x < img.cols) && (y >= 0 && y < img.rows);
	}

	bool HasColor(cv::Vec3b color)
	{
		return color[0] > 0 || color[1] > 0 || color[2] > 0;
	}

	int values()
	{
		return m;
	}
	
	int inputs()
	{
		return n;
	}

	// calculate error
	int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec)
	{
		// double gain1 = x[0];
		// double gain2 = x[1];
		
		double intensity1 = (avgItensity1[0] + avgItensity1[1] + avgItensity1[2]) / 3.0;
		double intensity2 = (avgItensity2[0] + avgItensity2[1] + avgItensity2[2]) / 3.0;
		double intensities[2] = { intensity1, intensity2 };
		for (int i = 0; i < values(); ++i)
		{
			/*cv::Vec3d diff = x[0] * avgItensity1 - x[1] * avgItensity2;
			double l2NormSqr = diff.ddot(diff);*/
			double diff = cv::min(x[i] * intensities[i], 255.0) - cv::min(x[(i + 1) % inputs()] * intensities[(i + 1) % inputs()], 255.0);
			fvec(i) = -0.5 * count *  ((diff * diff) / omegaN - (1 - x[i]) * (1 - x[i]) / omegaG);
		}

		return 0;
	}

	// calculate jacobian matrix
	int df(const Eigen::VectorXd& x, Eigen::MatrixXd& fjac)
	{
		double epsilon;
		epsilon = 1e-5f;
		for (int i = 0; i < x.size(); ++i)
		{
			Eigen::VectorXd xPlus(x);
			xPlus(i) += epsilon;

			Eigen::VectorXd xMinus(x);
			xMinus(i) -= epsilon;

			Eigen::VectorXd fvecPlus(values());
			operator()(xPlus, fvecPlus);

			Eigen::VectorXd fvecMinus(values());
			operator()(xMinus, fvecMinus);

			Eigen::VectorXd fvecDiff(values());
			fvecDiff = (fvecPlus - fvecMinus) / (2.0 * epsilon);

			fjac.block(0, i, values(), 1) = fvecDiff;
		}

		return 0;
	}
};

bool ContainImgID(const cv::Mat overlapMask, int x, int y, cv::uint8_t value)
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
				if (overlapMask.at<cv::uint8_t>(newY, newX) == value)
				{
					return true;
				}
			}
		}
	}
	return false;
}

int main(int argc, char* argv[])
{
	cv::Mat img1 = cv::imread("test4-result/warp_0.png", cv::IMREAD_COLOR);
	cv::Mat img2 = cv::imread("test4-result/warp_1.png", cv::IMREAD_COLOR);
	cv::Point2i offset1(0, 350), offset2(320, 374);

	if (!img1.empty() && !img2.empty())
	{

		int width = cv::max(offset1.x + img1.cols, offset2.x + img2.cols);
		int height = cv::max(offset1.y + img1.rows, offset2.y + img2.rows);
		cv::Rect rect1(offset1.x, offset1.y, img1.cols, img1.rows),
				rect2(offset2.x, offset2.y, img2.cols, img2.rows);

		cv::Rect opverlapRect(rect2.x, 0, rect1.x + rect1.width - rect2.x, height);
		cv::Mat overlapMask = cv::Mat::zeros(height, width, CV_8UC1);

		const cv::uint8_t imgID1 = 1, imgID2 = 2;
		const cv::uint8_t overlapID = imgID1 + imgID2;

		cv::Mat mask1 = CreateMask(img1, imgID1);
		cv::Mat mask2 = CreateMask(img2, imgID2);
		cv::add(overlapMask(rect1), mask1, overlapMask(rect1));
		cv::add(overlapMask(rect2), mask2, overlapMask(rect2));

		int minY = opverlapRect.height, maxY = 0;
		for (int row = 0; row < opverlapRect.height; ++row)
		{
			for (int col = 0; col < opverlapRect.width; ++col)
			{
				cv::uint8_t imgID = overlapMask.at<cv::uint8_t>(opverlapRect. y + row, opverlapRect.x + col);
				if (imgID == overlapID)
				{
					minY = cv::min(row, minY);
					maxY = cv::max(row, maxY);
				}
			}
		}
		opverlapRect.y = minY;
		opverlapRect.height = maxY - minY + 1;
		
		// gain compensation
		cv::Mat imgGray1, imgGray2;
		cv::cvtColor(img1, imgGray1, cv::COLOR_BGR2HSV);
		cv::cvtColor(img2, imgGray2, cv::COLOR_BGR2HSV);
		LMFunctor lmFunctor(imgGray1, imgGray2, opverlapRect, rect1, rect2);
		Eigen::LevenbergMarquardt<LMFunctor> lm(lmFunctor);

		Eigen::VectorXd x(2);
		x << 1.0, 1.0;
		lm.minimize(x);

		std::cout << x << std::endl;
		for (int row = 0; row < imgGray1.rows; ++row)
		{
			for (int col = 0; col < imgGray1.cols; ++col)
			{
				cv::Vec3b color = imgGray1.at<cv::Vec3b>(row, col);
				color[2] = cv::min(int(x[1] * (int)color[2]), 255);
				imgGray1.at<cv::Vec3b>(row, col) = color;
			}
		}

		for (int row = 0; row < imgGray2.rows; ++row)
		{
			for (int col = 0; col < imgGray2.cols; ++col)
			{
				cv::Vec3b color = imgGray2.at<cv::Vec3b>(row, col);
				color[2] = cv::min(int(x[1] * (int)color[2]), 255);
				imgGray2.at<cv::Vec3b>(row, col) = color;
			}
		}

		cv::cvtColor(imgGray1, img1, cv::COLOR_HSV2BGR);
		cv::cvtColor(imgGray2, img2, cv::COLOR_HSV2BGR);

		int numNode = 0;
		for (int row = 0; row < opverlapRect.height; ++row)
		{
			for (int col = 0; col < opverlapRect.width; ++col)
			{
				// check overlap
				if (overlapMask.at<cv::uint8_t>(opverlapRect.y + row, opverlapRect.x + col) != overlapID)
				{
					continue;
				}
				++numNode;
			}
		}

		//int numNode = opverlapRect.width * opverlapRect.height;
		//int numEdge = 2 * numNode - opverlapRect.width - opverlapRect.height;
		GCoptimizationGeneralGraph g(numNode, 2);
		cv::Mat lot = cv::Mat::zeros(opverlapRect.height, opverlapRect.width, CV_32S);

		auto edgeCost = [&img1, &img2, &opverlapRect, &rect1, &rect2](int x1, int y1, int x2, int y2) -> double
		{
			cv::Vec3b colorS1 = img1.at<cv::Vec3b>(opverlapRect.y + y1 - rect1.y, opverlapRect.x + x1 - rect1.x);
			cv::Vec3b colorT1 = img2.at<cv::Vec3b>(opverlapRect.y + y1 - rect2.y, opverlapRect.x + x1 - rect2.x);
			cv::Vec3i gradST1 = (cv::Vec3i)colorT1 - (cv::Vec3i)colorS1;

			cv::Vec3b colorS2 = img1.at<cv::Vec3b>(opverlapRect.y + y2 - rect1.y, opverlapRect.x + x2 - rect1.x);
			cv::Vec3b colorT2 = img2.at<cv::Vec3b>(opverlapRect.y + y2 - rect2.y, opverlapRect.x + x2 - rect2.x);
			cv::Vec3i gradST2 = (cv::Vec3i)colorT2 - (cv::Vec3i)colorS2;


			cv::Vec3i gradS = (cv::Vec3i)colorS2 - (cv::Vec3i)colorS1;
			cv::Vec3i gradT = (cv::Vec3i)colorT2 - (cv::Vec3i)colorT1;
			double l2Norm = cv::sqrt(gradST1[0] * gradST1[0] + gradST1[1] * gradST1[1] + gradST1[2] * gradST1[2]) + cv::sqrt(gradST2[0] * gradST2[0] + gradST2[1] * gradST2[1] + gradST2[2] * gradST2[2]);
			double l2Weight = cv::sqrt(gradS[0] * gradS[0] + gradS[1] * gradS[1] + gradS[2] * gradS[2]) + cv::sqrt(gradT[0] * gradT[0] + gradT[1] * gradT[1] + gradT[2] * gradT[2]);
			//return l2Norm;
			int gb = cv::abs(gradST1[0]) + cv::abs(gradST2[0]);
			int gg = cv::abs(gradST1[1]) + cv::abs(gradST2[1]);
			int gr = cv::abs(gradST1[2]) + cv::abs(gradST2[2]);
			double normalizeWeight = (cv::abs(gradS[0]) + cv::abs(gradS[1]) + cv::abs(gradS[2])) + (cv::abs(gradT[0]) + cv::abs(gradT[1]) + cv::abs(gradT[2]));

			return (gb + gg + gr) / 3;
		};

		std::vector<GCoptimization::SparseDataCost> dataCost1, dataCost2;

		try
		{
			int nodeIdx = -1;
			for (int row = 0; row < opverlapRect.height; ++row)
			{
				for (int col = 0; col < opverlapRect.width; ++col)
				{
					// check overlap
					if (overlapMask.at<cv::uint8_t>(opverlapRect.y + row, opverlapRect.x + col) != overlapID)
					{
						continue;
					}
					++nodeIdx;
					//std::cout << row << " " << col << " " << nodeIdx << " " << numNode << std::endl;
					lot.at<cv::int32_t>(row, col) = nodeIdx + 1;
					if (ContainImgID(overlapMask, opverlapRect.x + col, opverlapRect.y + row, imgID1))
					{
						// from left image
						dataCost1.push_back({ nodeIdx, 500 });
						dataCost2.push_back({ nodeIdx, 0 });
						//g.setDataCost(nodeIdx, 0, 500.0);
						//g.setDataCost(nodeIdx, std::numeric_limits<int>::max(), 0);

					}
					else if (ContainImgID(overlapMask, opverlapRect.x + col, opverlapRect.y + row, imgID2))
					{
						// from right image
						//g.add_tweights(nodeIdx, 0, std::numeric_limits<int>::max());
						dataCost1.push_back({ nodeIdx, 0 });
						dataCost2.push_back({ nodeIdx, 500 });
						//g.setDataCost(nodeIdx, 1, 500.0);
					}
					else
					{
						dataCost1.push_back({ nodeIdx, 0 });
						dataCost2.push_back({ nodeIdx, 0 });
						//g.setDataCost(nodeIdx, 0, 0.0);
						//g.setDataCost(nodeIdx, 1, 0.0);
					}

					// x direction
					if (col > 0 && overlapMask.at<cv::uint8_t>(opverlapRect.y + row, opverlapRect.x + col - 1) == overlapID)
					{
						double weight = edgeCost(col, row, col - 1, row);
						g.setNeighbors(nodeIdx, lot.at<cv::int32_t>(row, col - 1) - 1, weight);
						g.setNeighbors(lot.at<cv::int32_t>(row, col - 1) - 1, nodeIdx, weight);
					}

					// y direction
					if (row > 0 && overlapMask.at<cv::uint8_t>(opverlapRect.y + row - 1, opverlapRect.x + col) == overlapID)
					{
						double weight = edgeCost(col, row, col, row - 1);
						//g.add_edge(nodeIdx, lot.at<cv::int32_t>(row - 1, col) - 1, weight, weight);
						g.setNeighbors(nodeIdx, lot.at<cv::int32_t>(row - 1, col) - 1, weight);
						g.setNeighbors(lot.at<cv::int32_t>(row - 1, col) - 1, nodeIdx, weight);
					}
				}
			}
			g.setSmoothCost(0, 1, 1);
			g.setSmoothCost(0, 0, 0);
			g.setSmoothCost(1, 0, 1);
			g.setSmoothCost(1, 1, 0);

			g.setDataCost(0, dataCost1.data(), dataCost1.size());
			g.setDataCost(1, dataCost2.data(), dataCost2.size());

			std::cout << "E1: " << g.compute_energy() << std::endl;
			g.expansion(-1);
			std::cout << "E2: " << g.compute_energy() << std::endl;
		}
		catch (GCException e)
		{
			e.Report();
		}
		

		cv::Mat output(height, width, CV_8UC3, cv::Scalar::all(0));
		img1.copyTo(output(rect1), mask1);
		img2.copyTo(output(rect2), mask2);

		cv::Mat seamMask = cv::Mat::zeros(opverlapRect.height, opverlapRect.width, CV_8UC3);
		cv::Mat leftMask = cv::Mat::zeros(opverlapRect.height, opverlapRect.width, CV_8UC3),
				rightMask = cv::Mat::zeros(opverlapRect.height, opverlapRect.width, CV_8UC3);

		rightMask.setTo(cv::Scalar::all(1));
		leftMask.setTo(cv::Scalar::all(1));
		cv::Mat leftRGB = cv::Mat::zeros(opverlapRect.height, opverlapRect.width, CV_8UC3),
				rightRGB = cv::Mat::zeros(opverlapRect.height, opverlapRect.width, CV_8UC3);
		for (int row = 0; row < opverlapRect.height; ++row)
		{
			for (int col = 0; col < opverlapRect.width; ++col)
			{
				cv::uint8_t imgID = overlapMask.at<cv::uint8_t>(opverlapRect.y + row, opverlapRect.x + col);
				int nodeIdx = lot.at<cv::int32_t>(row, col);
				/*if (nodeIdx <= 0)
				{
					continue;
				}*/

				if ((nodeIdx > 0 && g.whatLabel(nodeIdx - 1) == 0) || imgID == imgID1)
				{
					seamMask.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 255);
					//leftMask.at<cv::Vec3b>(row, col) = cv::Vec3b(1, 1, 1);
					rightMask.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);
				}
				else if((nodeIdx > 0 && g.whatLabel(nodeIdx - 1) == 1) || imgID == imgID2)
				{
					seamMask.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 255, 0);
					leftMask.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);
					//rightMask.at<cv::Vec3b>(row, col) = cv::Vec3b(1, 1, 1);
				}

				if (rect1.contains(cv::Point2i(opverlapRect.x + col, opverlapRect.y + row)))
				{
					cv::Vec3b color1 = img1.at<cv::Vec3b>(opverlapRect.y + row - rect1.y, opverlapRect.x + col - rect1.x);
					leftRGB.at<cv::Vec3b>(row, col) = color1;
				}

				if (rect2.contains(cv::Point2i(opverlapRect.x + col , opverlapRect.y + row)))
				{
					cv::Vec3b color2 = img2.at<cv::Vec3b>(opverlapRect.y + row - rect2.y, opverlapRect.x + col - rect2.x);
					rightRGB.at<cv::Vec3b>(row, col) = color2;
				}
			}
		}

		
		MultiBandBlender blender;
		cv::Mat leftRGBF, rightRGBF;
		leftRGB.convertTo(leftRGBF, CV_32FC3, 1.0 / 255.0);
		rightRGB.convertTo(rightRGBF, CV_32FC3, 1.0 / 255.0);

		rightMask.convertTo(rightMask, CV_32FC3);
		cv::Mat res = blender.Blend(leftRGBF, rightRGBF, rightMask, 5);
		res.copyTo(output(opverlapRect), overlapMask(opverlapRect));

		// edge
		cv::Mat edgeMap;
		cv::cvtColor(leftMask * 255, edgeMap, cv::COLOR_BGR2GRAY);
		cv::Canny(edgeMap, edgeMap, 100, 200);

		cv::imwrite("output.png", output);
		output(opverlapRect).setTo(cv::Scalar(0, 0, 255), edgeMap);
		cv::imwrite("output_edge.png", output);


		cv::imshow("output", output);
		cv::imshow("output2", seamMask);
		cv::waitKey(0);

		/*cv::detail::MultiBandBlender blender2;
		blender2.feed(leftRGB, leftMask, cv::Point());
		blender2.feed(rightRGB, rightMask, cv::Point());
		blender2.blend(res, leftMask);*/

		
		//cv::imwrite("output.png", output);
	}
	else
	{
		std::cout << "image read failed" << std::endl;
	}

	return 0;
}