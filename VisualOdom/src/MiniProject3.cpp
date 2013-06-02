#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "FrameData.h"
#include <fstream>

using namespace cv;
using namespace std;

void compute3DPoints(const Mat& imgLeft, const Mat& imgRight, FrameData& result)
{
	//-- Step 1: Detect the keypoints using SURF Detector
		int minHessian = 400;

		SurfFeatureDetector detector( minHessian );

		std::vector<KeyPoint> keypoints_1, keypoints_2;

		detector.detect( imgLeft, keypoints_1 );
		detector.detect( imgRight, keypoints_2 );

		//-- Step 2: Calculate descriptors (feature vectors)
		SurfDescriptorExtractor extractor;

		Mat descriptors_1, descriptors_2;

		extractor.compute( imgLeft, keypoints_1, descriptors_1 );
		extractor.compute( imgRight, keypoints_2, descriptors_2 );

		#ifdef DEBUG
			std::cout << "Size of descriptor1: " << descriptors_1.size() << std::endl;
			std::cout << "Size of descriptor2: " << descriptors_2.size() << std::endl;
		#endif

		//-- Step 3: Matching descriptor vectors using FLANN matcher
		FlannBasedMatcher matcher;
		std::vector< DMatch > matches;
		matcher.match( descriptors_1, descriptors_2, matches );

		double max_dist = 0; double min_dist = 100;

		//-- Quick calculation of max and min distances between keypoints
		for( int i = 0; i < descriptors_1.rows; i++ )
		{ double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
		}

		#ifdef DEBUG
			cout << "-- Max dist : " <<  max_dist << endl;
			cout << "-- Max dist : " <<  max_dist << endl;
		#endif

		//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
		//-- PS.- radiusMatch can also be used here.
		std::vector< DMatch > good_matches;
		int scale = 2;

		while( good_matches.size() < 100)
		{
			good_matches.clear();
			for( int i = 0; i < descriptors_1.rows; i++ )
			{
				if( matches[i].distance < scale*min_dist )                        //< 2*min_dist
				{
					good_matches.push_back( matches[i]);
				}
			}
			scale += 1;
		}


		//-- Draw only "good" matches
		Mat img_matches;
		drawMatches( imgLeft, keypoints_1, imgRight, keypoints_2,
			   good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			   vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

		//-- Show detected matches
		#ifdef DEBUG
			imshow( "Good Matches", img_matches );
			waitKey(0);
		#endif

//		for( int i = 0; i < good_matches.size(); i++ )
//		{ printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

		Mat left_matched_points_mat(2,good_matches.size(),CV_32F);
		Mat right_matched_points_mat(2,good_matches.size(),CV_32F);

		Mat goodDescriptors(0,good_matches.size(),CV_32F);
		for( int i = 0; i < good_matches.size(); i++)
		{
			left_matched_points_mat.at<float>(0,i) = keypoints_1[ good_matches[i].queryIdx ].pt.x;
			left_matched_points_mat.at<float>(1,i) = keypoints_1[ good_matches[i].queryIdx ].pt.y;
			right_matched_points_mat.at<float>(0,i) = keypoints_2[ good_matches[i].trainIdx ].pt.x;
			right_matched_points_mat.at<float>(1,i) = keypoints_2[ good_matches[i].trainIdx ].pt.y;
			goodDescriptors.push_back(descriptors_1.row(good_matches[i].queryIdx));

		}
		result.descriptors = goodDescriptors;

		triangulatePoints(projectionLeft,projectionRight,left_matched_points_mat,right_matched_points_mat,result.points3d);

		for(int i = 0; i < good_matches.size(); i++)
		{
			result.points3d.at<float>(0,i) /= result.points3d.at<float>(3,i);
			result.points3d.at<float>(1,i) /= result.points3d.at<float>(3,i);
			result.points3d.at<float>(2,i) /= result.points3d.at<float>(3,i);
			result.points3d.at<float>(3,i) /= result.points3d.at<float>(3,i);
		}
}

void writePointsToFile(const vector<Point3d>& points, string fileName)
{
	ofstream file;
	file.open(fileName.c_str());

	for(int i = 0; i < points.size(); i++)
	{
		  file << points[i].x << "," << points[i].y << "," << points[i].z << endl;
	}

	file.close();
}

void computeCentroids(const std::vector<Point3d>& points, Point3d& mean)
{
	double sumX = 0, sumY = 0, sumZ = 0;

		for(int i = 0; i < points.size(); i++)
		{
			sumX += points[i].x;
			sumY += points[i].y;
			sumZ += points[i].z;
		}
		mean.x = sumX / points.size();
		mean.y = sumY / points.size();
		mean.z = sumZ / points.size();
}

void demeanPoints(const vector<Point3d>& src, vector<Point3d>& dst, const Point3d& mean)
{
	Point3f temp;
	for(int i = 0; i < src.size(); i++)
	{
		temp.x = src[i].x - mean.x;
		temp.y = src[i].y - mean.y;
		temp.z = src[i].z - mean.z;
		dst.push_back(temp);
	}
}

void computeMatches(const FrameData& oldFrame, const FrameData& newFrame, vector<Point3d>& oldPoints, vector<Point3d>& newPoints )
{
	//-- Step 3: Matching descriptor vectors using FLANN matcher
		FlannBasedMatcher matcher;
		std::vector< DMatch > matches;
		matcher.match( oldFrame.descriptors, newFrame.descriptors, matches );

		double max_dist = 0; double min_dist = 100;

		//-- Quick calculation of max and min distances between keypoints
		for( int i = 0; i < oldFrame.descriptors.rows; i++ )
		{ double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
		}

		//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
		//-- PS.- radiusMatch can also be used here.
		std::vector< DMatch > good_matches;
		int scale = 2;

		while( good_matches.size() < 30)
		{
			good_matches.clear();
			for( int i = 0; i < oldFrame.descriptors.rows; i++ )
			{ if( matches[i].distance < scale*min_dist )                        //< 2*min_dist
			{ good_matches.push_back( matches[i]); }
			}
			scale += 1;
		}

		Mat matchingPointsLast(3,good_matches.size(),CV_32F);
		Mat matchingPointsNew(3,good_matches.size(),CV_32F);

		for(int i = 0; i < good_matches.size(); i++)
		{
				Point3f temp;
				temp.x = oldFrame.points3d.at<float>(0,good_matches[i].queryIdx);
				temp.y = oldFrame.points3d.at<float>(1,good_matches[i].queryIdx);
				temp.z = oldFrame.points3d.at<float>(2,good_matches[i].queryIdx);
				oldPoints.push_back(temp);
				temp.x = newFrame.points3d.at<float>(0,good_matches[i].trainIdx);
				temp.y = newFrame.points3d.at<float>(1,good_matches[i].trainIdx);
				temp.z = newFrame.points3d.at<float>(2,good_matches[i].trainIdx);
				newPoints.push_back(temp);
		}
		#ifdef DEBUG
			cout << "Number of matching 3D points: " << newPoints.size() << endl << endl;
		#endif
}

void computeCorrelationMatrix(const vector<Point3d>& oldPointsDemean, const vector<Point3d>& newPointsDemean, Mat& corrMat)
{
	Mat temp(3,3,CV_64F);

	for(int i = 0; i < 3; i++)
		{
			for(int j = 0; j < 3; j++)
			{
				temp.at<double>(i,j) = 0;

				for(int k = 0; k < oldPointsDemean.size(); k++)
				{
					if(i == 0)
						if(j == 0)
							temp.at<double>(i,j) += oldPointsDemean[k].x * newPointsDemean[k].x;
						else if(j == 1)
							temp.at<double>(i,j) += oldPointsDemean[k].x * newPointsDemean[k].y;
						else if(j == 2)
							temp.at<double>(i,j) += oldPointsDemean[k].x * newPointsDemean[k].z;

					if(i == 1)
						if(j == 0)
							temp.at<double>(i,j) += oldPointsDemean[k].y * newPointsDemean[k].x;
						else if(j == 1)
							temp.at<double>(i,j) += oldPointsDemean[k].y * newPointsDemean[k].y;
						else if(j == 2)
							temp.at<double>(i,j) += oldPointsDemean[k].y * newPointsDemean[k].z;

					if(i == 2)
						if(j == 0)
							temp.at<double>(i,j) += oldPointsDemean[k].z * newPointsDemean[k].x;
						else if(j == 1)
							temp.at<double>(i,j) += oldPointsDemean[k].z * newPointsDemean[k].y;
						else if(j == 2)
							temp.at<double>(i,j) += oldPointsDemean[k].z * newPointsDemean[k].z;
				}
			}
		}
	corrMat = temp;
}

void computeRotation(const Mat& corrMat, Mat& R1, Mat& R2)
{
	SVD decomp = SVD(corrMat);
	R1 = decomp.u * decomp.vt;			// Orthogonal Procrustes algorithm
	R2 = decomp.vt.t() * decomp.u.t();	// Matlab code
}

void computeTranslation(const Point3d& oldCentroid, const Point3d& newCentroid, const Mat& R, Point3d& T1, Point3d& T2)
{
	Point3d oldCentroidRot;
	oldCentroidRot.x = oldCentroid.x * R.at<double>(0,0) + oldCentroid.y * R.at<double>(0,1) + oldCentroid.z * R.at<double>(0,2);
	oldCentroidRot.y = oldCentroid.x * R.at<double>(1,0) + oldCentroid.y * R.at<double>(1,1) + oldCentroid.z * R.at<double>(1,2);
	oldCentroidRot.z = oldCentroid.x * R.at<double>(2,0) + oldCentroid.y * R.at<double>(2,1) + oldCentroid.z * R.at<double>(2,2);

	T1 = newCentroid - oldCentroidRot;
	T2 = newCentroid - oldCentroid;
}

void transformPoints(const vector<Point3d>& inPoints, const Mat& R, const Point3d& T, vector<Point3d>& outPoints)
{
	for(int i = 0; i < inPoints.size(); i++)
	{
		Point3d temp;
		temp.x = (R.at<double>(0,0)*inPoints[i].x + R.at<double>(0,1)*inPoints[i].y + R.at<double>(0,2)*inPoints[i].z) + T.x;
		temp.y = (R.at<double>(1,0)*inPoints[i].x + R.at<double>(1,1)*inPoints[i].y + R.at<double>(1,2)*inPoints[i].z) + T.y;
		temp.z = (R.at<double>(2,0)*inPoints[i].x + R.at<double>(2,1)*inPoints[i].y + R.at<double>(2,2)*inPoints[i].z) + T.z;
		outPoints.push_back(temp);
	}
}

int computeInliers(const vector<Point3d>& oldPointsTransformed, const vector<Point3d>& newPoints, vector<int>& inlierIndexes, double& avgError)
{
	int inliers = 0;
	double sum = 0;
	for(int i = 0; i < newPoints.size(); i++)
	{
		Point3d diff = newPoints[i] - oldPointsTransformed[i];
		double euclidean = sqrt(pow(diff.x,2.0) + pow(diff.y,2.0) + pow(diff.z,2.0));
		if(euclidean < 0.02)
		{
			inlierIndexes.push_back(i);
			inliers++;
			sum += euclidean;
		}
	}
	avgError = sum/inlierIndexes.size();

	return inliers;
}

void computeTransform(const vector<Point3d>& oldPoints, const vector<Point3d>& newPoints, Mat& R, Point3d& T)
{
	Point3d oldCentroid, newCentroid;
	computeCentroids(oldPoints, oldCentroid);
	computeCentroids(newPoints, newCentroid);

	vector<Point3d> oldPointsDemean, newPointsDemean;
	demeanPoints(oldPoints, oldPointsDemean, oldCentroid);
	demeanPoints(newPoints, newPointsDemean, newCentroid);

	Mat corrMat(3,3,CV_64F);
	computeCorrelationMatrix(oldPointsDemean, newPointsDemean, corrMat);

	Mat R1, R2;
	computeRotation(corrMat, R1, R2);

	Point3d T1, T2;
	computeTranslation(oldCentroid, newCentroid, R2, T1, T2);

	R = R2;
	T = T1;

	#ifdef DEBUG
		cout << "Old centroid: " << oldCentroid << endl;
		cout << "New centroid: " << newCentroid << endl << endl;
		cout << "Correlation matrix: " << corrMat<< endl << endl;
		cout << "Rotation1: " << R1 << endl;
		cout << "Rotation2: " << R2 << endl << endl;
		cout << "Translation1: " << T1 << endl;
		cout << "Translation2: " << T2 << endl << endl;
	#endif
}

void getRandomPoints(const vector<Point3d>& oldPoints, const vector<Point3d>& newPoints, vector<Point3d>& randomOldPoints, vector<Point3d>& randomNewPoints)
{
	vector<int> usedIndex;
	while(randomNewPoints.size() < 3)
	{
		bool newIndex = true;
		int randomIndex = rand()%oldPoints.size();
		for(int i = 0; i < usedIndex.size(); i++)
			if(randomIndex == usedIndex[i])
				newIndex = false;

		if(newIndex)
		{
			randomOldPoints.push_back(oldPoints[randomIndex]);
			randomNewPoints.push_back(newPoints[randomIndex]);
			usedIndex.push_back(randomIndex);
		}
	}
}

void computeTransformation(const FrameData& oldFrame, FrameData& newFrame)
{
	// Match features
	vector<Point3d> oldPoints, newPoints;
	computeMatches(oldFrame, newFrame, oldPoints, newPoints);

	// Initialize parameters for RANSAC
	vector<int> bestInlierIndexes;
	int maxInliers = 0;
	Mat bestRotation;
	Point3d bestTranslation;
	double bestAvgError;
	int maxRuns = 100;

	for(int run = 0; run < maxRuns; run++)
	{
		// Choose random points
		vector<Point3d> randomOldPoints, randomNewPoints;
		getRandomPoints(oldPoints, newPoints, randomOldPoints, randomNewPoints);

		// Compute transformation
		Mat R;
		Point3d T;
		computeTransform(randomOldPoints, randomNewPoints, R, T);

		// Apply transform
		vector<Point3d> oldPointsTransformed;
		transformPoints(oldPoints, R, T, oldPointsTransformed);

		// Count inliers
		vector<int> inlierIndexes;
		double avgError;
		int inliers = computeInliers(oldPointsTransformed, newPoints, inlierIndexes, avgError);

		// Update max inliers
		if(inliers > maxInliers)
		{
			bestInlierIndexes = inlierIndexes;
			maxInliers = inliers;
			bestAvgError = avgError;
			bestRotation = R;
			bestTranslation = T;
		}
	}
	cout << endl << "!!!!!!!!!!!!! RANSAC RESULT !!!!!!!!!!!!!" << endl;
	cout << "Max inliers: " << maxInliers << endl;
	cout << "Best rotation: " << bestRotation << endl;
	cout << "Best translation: " << bestTranslation << endl;
	cout << "Average euclidean distance: " << bestAvgError << endl;

	// Reestimate tranform based on inliers
	vector<Point3d> oldPointInliers, newPointInliers;
	for(int i = 0; i < bestInlierIndexes.size(); i++)
	{
		oldPointInliers.push_back(oldPoints[bestInlierIndexes[i]]);
		newPointInliers.push_back(newPoints[bestInlierIndexes[i]]);
	}

	computeTransform(oldPointInliers, newPointInliers, bestRotation, bestTranslation);

	vector<Point3d> oldPointsInliersTransformed;
	transformPoints(oldPointInliers, bestRotation, bestTranslation, oldPointsInliersTransformed);

	vector<int> inlierIndexes;
	int inliers = computeInliers(oldPointsInliersTransformed, newPointInliers, inlierIndexes, bestAvgError);
	cout << endl << "!!!!!!!!!!!!! REESTIMATED RESULT !!!!!!!!!!!!!" << endl;
	cout << "Max inliers: " << inliers << endl;
	cout << "New rotation: " << bestRotation << endl;
	cout << "New translation: " << bestTranslation << endl;
	cout << "Average euclidean distance: " << bestAvgError << endl;

	for(int row = 0; row < 3; row++)
	{
		for(int col = 0; col < 3; col++)
		{
			newFrame.transform.at<double>(row,col) = bestRotation.at<double>(row,col);
		}
	}
	newFrame.transform.at<double>(0,3) = bestTranslation.x;
	newFrame.transform.at<double>(1,3) = bestTranslation.y;
	newFrame.transform.at<double>(2,3) = bestTranslation.z;
}

/** @function main */
int main( int argc, char** argv )
{
	FileStorage fsLeft("left.yaml", FileStorage::READ);
	Mat intrinsicsLeft, distortionLeft, rectificationLeft;
	fsLeft["camera_matrix"] >> intrinsicsLeft;
	fsLeft["distortion_coefficients"] >> distortionLeft;
	fsLeft["rectification_matrix"] >> rectificationLeft;
	fsLeft["projection_matrix"] >> projectionLeft;

	FileStorage fsRight("right.yaml", FileStorage::READ);
	Mat intrinsicsRight, distortionRight, rectificationRight;
	fsRight["camera_matrix"] >> intrinsicsRight;
	fsRight["distortion_coefficients"] >> distortionRight;
	fsRight["rectification_matrix"] >> rectificationRight;
	fsRight["projection_matrix"] >> projectionRight;

	Mat img_l1 = imread( "images/test3left1.jpg", CV_LOAD_IMAGE_GRAYSCALE );
	Mat img_r1 = imread( "images/test3right1.jpg", CV_LOAD_IMAGE_GRAYSCALE );
	Mat img_l2 = imread( "images/test3left2.jpg", CV_LOAD_IMAGE_GRAYSCALE );
	Mat img_r2 = imread( "images/test3right2.jpg", CV_LOAD_IMAGE_GRAYSCALE );

	if( !img_l1.data || !img_r1.data || !img_l2.data || !img_r2.data )
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }

	std::vector<FrameData> data;
	FrameData tempData1;
	FrameData tempData2;

	compute3DPoints(img_l1, img_r1, tempData1);

	cout << tempData1.points3d << endl;

	compute3DPoints(img_l2, img_r2, tempData2);

	cout << tempData2.points3d << endl;

	computeTransformation(tempData1, tempData2);

	cout << "List of transforms:" << endl;
	cout << tempData1.transform << endl;
	cout << tempData2.transform << endl;

	return 0;
 }
