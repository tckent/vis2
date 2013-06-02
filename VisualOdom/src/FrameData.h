/********************************************************************************************************************
 *
 * @file		FrameData.h
 * @author		Kent Hansen (kenth09@student.sdu.dk)
 * @date		2013-05-23
 * @version		1.0
 * @brief		Class for FrameData type
 *
*********************************************************************************************************************/

#ifndef FRAMEDATA_HPP_
#define FRAMEDATA_HPP_

#include "opencv2/opencv.hpp"

using namespace cv;

class FrameData
{
	public:
		/// Constructor
		FrameData() { T1 = Mat::eye(4,4,CV_64F); T2 = Mat::eye(4,4,CV_64F); };

		Mat descriptors;
		Mat allPoints3d;
		Mat points3d;
		Mat T1;
		Mat T2;

	private:

};

#endif /* FRAMEDATA_HPP_ */
