#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud.h>
#include <tf/transform_broadcaster.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <ros/package.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "FrameData.h"
#include <fstream>

using namespace cv;
using namespace std;
namespace enc = sensor_msgs::image_encodings;

// Global variables
cv_bridge::CvImagePtr right_image_ptr;
cv_bridge::CvImagePtr left_image_ptr;

bool rightImageExists(false);
bool leftImageExists(false);
bool firstRun(true);

ros::Publisher odom_pub;
ros::Publisher point_cloud_pub;
Mat oldTransform;
ros::Time lastTime;
ros::Time currentTime;
Mat projectionLeft, projectionRight;
FrameData oldFrame;
FrameData newFrame;
int frameCount;
int errorCount;

vector<Point3d> transforms;

//#define DEBUG


void compute3DPoints(const Mat& imgLeft, const Mat& imgRight, FrameData& result)
{
	//-- Step 1: Detect the keypoints using SURF Detector
		int minHessian = 400;
		double tmpTime = ros::Time::now().toSec();

		SurfFeatureDetector detector( minHessian );

		std::vector<KeyPoint> keypoints_1, keypoints_2;

		detector.detect( imgLeft, keypoints_1 );
		detector.detect( imgRight, keypoints_2 );

		cout << "compute3DPoints: SurfFeatureDection for both images: " << (ros::Time::now().toSec() - tmpTime) << endl;
		tmpTime = ros::Time::now().toSec();

		//-- Step 2: Calculate descriptors (feature vectors)
		SurfDescriptorExtractor extractor;

		Mat descriptors_1, descriptors_2;

		extractor.compute( imgLeft, keypoints_1, descriptors_1 );
		extractor.compute( imgRight, keypoints_2, descriptors_2 );

		#ifdef DEBUG
			std::cout << "Size of descriptor1: " << descriptors_1.size() << std::endl;
			std::cout << "Size of descriptor2: " << descriptors_2.size() << std::endl;
		#endif

		cout << "compute3DPoints: SurfExtractor for both images: " << (ros::Time::now().toSec() - tmpTime) << endl;
		tmpTime = ros::Time::now().toSec();
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

		int minGoodMatches = 100;

		if (matches.size() < minGoodMatches)
		{
			minGoodMatches = matches.size();
		}
		while( good_matches.size() < minGoodMatches)
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

		cout << "compute3DPoints: FeatureMatching for both images: " << (ros::Time::now().toSec() - tmpTime) << endl;
		tmpTime = ros::Time::now().toSec();
		#ifdef DEBUG
			//-- Draw only "good" matches
			Mat img_matches;
			drawMatches( imgLeft, keypoints_1, imgRight, keypoints_2,
				   good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
				   vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

			//-- Show detected matches

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

		cout << "compute3DPoints: Triangulate points: " << (ros::Time::now().toSec() - tmpTime) << endl;
		tmpTime = ros::Time::now().toSec();
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

		int minGoodMatches = 30;
		if (matches.size() < minGoodMatches)
		{
			minGoodMatches = matches.size();
		}
		while( good_matches.size() < minGoodMatches)
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

void computeTransform(const vector<Point3d>& oldPoints, const vector<Point3d>& newPoints, Mat& R, Point3d& T, bool withRotation)
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
	if(withRotation)
		T = T1;
	else
		T = T2;

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

void computeTransformation(FrameData& oldFrame, FrameData& newFrame)
{
	double tmpTime = ros::Time::now().toSec();
	// Match features
	vector<Point3d> oldPoints, newPoints;
	computeMatches(oldFrame, newFrame, oldPoints, newPoints);

	cout << "computeTransformation: computeMatches: " << (ros::Time::now().toSec() - tmpTime) << endl;
	tmpTime = ros::Time::now().toSec();
	// Initialize parameters for RANSAC
	vector<int> bestInlierIndexes;
	int maxInliers = 0;
	Mat bestRotation;
	Point3d bestTranslation;
	double bestAvgError;
	int maxRuns = 5000;

	for(int run = 0; run < maxRuns; run++)
	{
		// Choose random points
		vector<Point3d> randomOldPoints, randomNewPoints;
		getRandomPoints(oldPoints, newPoints, randomOldPoints, randomNewPoints);

		// Compute transformation
		Mat R;
		Point3d T;
		computeTransform(randomNewPoints, randomOldPoints, R, T, true);

		// Apply transform
		vector<Point3d> newPointsTransformed;
		transformPoints(newPoints, R, T, newPointsTransformed);

		// Count inliers
		vector<int> inlierIndexes;
		double avgError;
		int inliers = computeInliers(newPointsTransformed, oldPoints, inlierIndexes, avgError);

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


	cout << "computeTransformation: Ransac: " << (ros::Time::now().toSec() - tmpTime) << endl;
	tmpTime = ros::Time::now().toSec();

	// Reestimate tranform based on inliers
	vector<Point3d> oldPointInliers, newPointInliers;
	for(int i = 0; i < bestInlierIndexes.size(); i++)
	{
		oldPointInliers.push_back(oldPoints[bestInlierIndexes[i]]);
		newPointInliers.push_back(newPoints[bestInlierIndexes[i]]);
	}

	computeTransform(newPointInliers, oldPointInliers, bestRotation, bestTranslation, true);

	Point3d translationWithoutRot;
	computeTransform(newPointInliers, oldPointInliers, bestRotation, translationWithoutRot, false);

	vector<Point3d> newPointsInliersTransformed;
	transformPoints(newPointInliers, bestRotation, bestTranslation, newPointsInliersTransformed);

	vector<int> inlierIndexes;
	int inliers = computeInliers(newPointsInliersTransformed, oldPointInliers, inlierIndexes, bestAvgError);
	cout << endl << "!!!!!!!!!!!!! REESTIMATED RESULT !!!!!!!!!!!!!" << endl;
	cout << "Max inliers: " << inliers << endl;
	cout << "New rotation: " << bestRotation << endl;
	cout << "New translation: " << bestTranslation << endl;
	cout << "New trans w/o rot: " << translationWithoutRot << endl;
	cout << "Average euclidean distance: " << bestAvgError << endl;

	for(int row = 0; row < 3; row++)
	{
		for(int col = 0; col < 3; col++)
		{
			newFrame.T1.at<double>(row,col) = bestRotation.at<double>(row,col);
			newFrame.T2.at<double>(row,col) = bestRotation.at<double>(row,col);

		}
	}
	cout << "computeTransformation: Reestimate: " << (ros::Time::now().toSec() - tmpTime) << endl;
	tmpTime = ros::Time::now().toSec();

	oldFrame.points3d = Mat(3,newPointInliers.size(),CV_32F);
	for (int i = 0; i < newPointInliers.size(); i++)
	{
		oldFrame.points3d.at<float>(0,i) = newPointInliers.at(i).x;
		oldFrame.points3d.at<float>(1,i) = newPointInliers.at(i).y;
		oldFrame.points3d.at<float>(2,i) = newPointInliers.at(i).z;
	}
	newFrame.T1.at<double>(0,3) = bestTranslation.x;
	newFrame.T1.at<double>(1,3) = bestTranslation.y;
	newFrame.T1.at<double>(2,3) = bestTranslation.z;
	newFrame.T2.at<double>(0,3) = translationWithoutRot.x;
	newFrame.T2.at<double>(1,3) = translationWithoutRot.y;
	newFrame.T2.at<double>(2,3) = translationWithoutRot.z;
}

void publishPointCloud(const Mat& points, bool error)
{
	sensor_msgs::PointCloud pc;
	pc.header.stamp = currentTime;
	pc.header.frame_id = "base_link";
	sensor_msgs::ChannelFloat32 intensity;
	intensity.name = "Intensity";
	vector<float> values;

	for (int i = 0; i < points.cols; i++)
	{
		geometry_msgs::Point32 tmpPoint;
		tmpPoint.x = points.at<float>(0,i);
		tmpPoint.y = points.at<float>(1,i);
		tmpPoint.z = points.at<float>(2,i);
		intensity.values.push_back(255);
		pc.points.push_back(tmpPoint);
	}

	geometry_msgs::Point32 tmpPoint;
	tmpPoint.x = 0;
	tmpPoint.y = 0;
	tmpPoint.z = 0;
	if (error)
		intensity.values.push_back(1);
	else
		intensity.values.push_back(100);

	pc.points.push_back(tmpPoint);
	pc.channels.push_back(intensity);
	point_cloud_pub.publish(pc);
}

void publishTransformAndOdometry(Mat transform)
{
	if (transform.cols != 4 || transform.rows != 4)
	{
		ROS_INFO("Transform matrix has wrong dimentions");
		return;
	}

    oldTransform = oldTransform * transform;

    Mat tmpTrans = Mat::zeros(4,4,CV_64F);
    tmpTrans.at<double>(3,3) = 1;
    tmpTrans.at<double>(0,2) = -1;
    tmpTrans.at<double>(2,0) = 1;
    tmpTrans.at<double>(1,1) = 1;

    //tmpTrans = oldTransform * tmpTrans;
    tmpTrans = oldTransform;
	static tf::TransformBroadcaster br;

	//  Calculate quaternion
	geometry_msgs::Quaternion oldQuat;
	oldQuat.w = sqrt(1.0 + tmpTrans.at<double>(0,0) + tmpTrans.at<double>(1,1) + tmpTrans.at<double>(2,2)) / 2.0;
	oldQuat.x = (tmpTrans.at<double>(2,1) - tmpTrans.at<double>(1,2)) / (4.0*oldQuat.w);
	oldQuat.y = (tmpTrans.at<double>(0,2) - tmpTrans.at<double>(2,0)) / (4.0*oldQuat.w);
	oldQuat.z = (tmpTrans.at<double>(1,0) - tmpTrans.at<double>(0,1)) / (4.0*oldQuat.w);

	// Publish tf oldTransform
	geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = currentTime;
    odom_trans.header.frame_id = "world";
    odom_trans.child_frame_id = "base_link";

    odom_trans.transform.translation.x = tmpTrans.at<double>(0,3);
    odom_trans.transform.translation.y = tmpTrans.at<double>(1,3);
    odom_trans.transform.translation.z = tmpTrans.at<double>(2,3);
    odom_trans.transform.rotation = oldQuat;
    br.sendTransform(odom_trans);

	// publish Odometry message
    nav_msgs::Odometry odom;
    odom.header.stamp = currentTime;
    odom.header.frame_id = "world";

    //set the position
    odom.pose.pose.position.x = tmpTrans.at<double>(0,3);
    odom.pose.pose.position.y = tmpTrans.at<double>(1,3);
    odom.pose.pose.position.z = tmpTrans.at<double>(2,3);
    odom.pose.pose.orientation.w = oldQuat.w;
    odom.pose.pose.orientation.x = oldQuat.x;
    odom.pose.pose.orientation.y = oldQuat.y;
    odom.pose.pose.orientation.z = oldQuat.z;

    //set the velocity
    odom.child_frame_id = "base_link";
    odom.twist.twist.linear.x = transform.at<double>(0,3) / (currentTime - lastTime).toSec();
    odom.twist.twist.linear.y = transform.at<double>(1,3) / (currentTime - lastTime).toSec();
    odom.twist.twist.angular.z = transform.at<double>(2,3) / (currentTime - lastTime).toSec();

    //publish the message
    odom_pub.publish(odom);

    Point3d tmp;
    tmp.x = oldTransform.at<double>(0,3);
    tmp.y = oldTransform.at<double>(1,3);
    tmp.z = oldTransform.at<double>(2,3);

    transforms.push_back(tmp);

    lastTime = currentTime;
}

void publishTransformAndOdometry2(Mat transform)
{
	if (transform.cols != 4 || transform.rows != 4)
	{
		ROS_INFO("Transform matrix has wrong dimentions");
		return;
	}

	static tf::TransformBroadcaster br;

	//  Calculate quaternion
	geometry_msgs::Quaternion oldQuat;
	oldQuat.w = sqrt(1.0 + oldTransform.at<double>(0,0) + oldTransform.at<double>(1,1) + oldTransform.at<double>(2,2)) / 2.0;
	oldQuat.x = (oldTransform.at<double>(2,1) - oldTransform.at<double>(1,2)) / (4.0*oldQuat.w);
	oldQuat.y = (oldTransform.at<double>(0,2) - oldTransform.at<double>(2,0)) / (4.0*oldQuat.w);
	oldQuat.z = (oldTransform.at<double>(1,0) - oldTransform.at<double>(0,1)) / (4.0*oldQuat.w);

	// Publish tf oldTransform
	geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = currentTime;
    odom_trans.header.frame_id = "world";
    odom_trans.child_frame_id = "base_link";

    odom_trans.transform.translation.x = oldTransform.at<double>(0,3);
    odom_trans.transform.translation.y = oldTransform.at<double>(1,3);
    odom_trans.transform.translation.z = oldTransform.at<double>(2,3);
    odom_trans.transform.rotation = oldQuat;
    br.sendTransform(odom_trans);

    // Calculate new quaternion
	geometry_msgs::Quaternion newQuat;
	newQuat.w = sqrt(1.0 + transform.at<double>(0,0) + transform.at<double>(1,1) + transform.at<double>(2,2)) / 2.0;
	newQuat.x = (transform.at<double>(2,1) - transform.at<double>(1,2)) / (4.0*newQuat.w);
	newQuat.y = (transform.at<double>(0,2) - transform.at<double>(2,0)) / (4.0*newQuat.w);
	newQuat.z = (transform.at<double>(1,0) - transform.at<double>(0,1)) / (4.0*newQuat.w);

	// publish Odometry message
    nav_msgs::Odometry odom;
    odom.header.stamp = currentTime;
    odom.header.frame_id = "base_link";

    //set the position
    odom.pose.pose.position.x = transform.at<double>(0,3);
    odom.pose.pose.position.y = transform.at<double>(1,3);
    odom.pose.pose.position.z = transform.at<double>(2,3);
    odom.pose.pose.orientation.w = newQuat.w;
    odom.pose.pose.orientation.x = newQuat.x;
    odom.pose.pose.orientation.y = newQuat.y;
    odom.pose.pose.orientation.z = newQuat.z;

    //set the velocity
    odom.child_frame_id = "odom";
    odom.twist.twist.linear.x = transform.at<double>(0,3) / (currentTime - lastTime).toSec();
    odom.twist.twist.linear.y = transform.at<double>(1,3) / (currentTime - lastTime).toSec();
    odom.twist.twist.angular.z = transform.at<double>(2,3) / (currentTime - lastTime).toSec();

    //publish the message
    odom_pub.publish(odom);

    // Update old transform
    oldTransform = oldTransform * transform;

    Point3d tmp;
    tmp.x = oldTransform.at<double>(0,3);
    tmp.y = oldTransform.at<double>(1,3);
    tmp.z = oldTransform.at<double>(2,3);

    transforms.push_back(tmp);

    lastTime = currentTime;
}

void processStereoImages()
{
	//imshow("LeftImage", left_image_ptr->image);
	//imshow("RightImage", right_image_ptr->image);
	//cv::waitKey(3);
	double tmpTime = ros::Time::now().toSec();
	double startTime = ros::Time::now().toSec();

	compute3DPoints(left_image_ptr->image,right_image_ptr->image,newFrame);

	cout << "Total time compute3DPoints: " << (ros::Time::now().toSec() - tmpTime) << endl;
	tmpTime = ros::Time::now().toSec();
	if (firstRun)
		firstRun = false;
	else
	{
		computeTransformation(oldFrame, newFrame);
		cout << "Total time computeTransformation: " << (ros::Time::now().toSec() - tmpTime) << endl;
		tmpTime = ros::Time::now().toSec();
		double translation = sqrt(pow(newFrame.T1.at<double>(0,3),2.0) + pow(newFrame.T1.at<double>(1,3),2.0) + pow(newFrame.T1.at<double>(2,3),2.0));
		cout << "Translation norm2: " << translation << endl;
		if (translation < 0.1) {
			publishTransformAndOdometry(newFrame.T1);
			publishPointCloud(oldFrame.points3d, false);
		}
		else {
			cout << "Transform discarded - total errors: " << ++errorCount << endl;
			//publishTransformAndOdometry(oldFrame.T1);
			//Mat tmp = Mat::zeros(3,1,CV_32F);
			//publishPointCloud(tmp, true);

		}
	}
	oldFrame = newFrame;

	if (transforms.size() % 10 == 0)
		writePointsToFile(transforms, "transforms.txt");

	cout << "Total frame: " << (ros::Time::now().toSec() - startTime) << endl;

}

//This function is called everytime a new image is published
void leftImageCallback(const sensor_msgs::ImageConstPtr& original_image)
{
	//Convert from the ROS image message to a CvImage suitable for working with OpenCV for processing
	try
	{
		//Always copy, returning a mutable CvImage
		//OpenCV expects color images to use BGR channel order.
		left_image_ptr = cv_bridge::toCvCopy(original_image, enc::MONO8);
	}
	catch (cv_bridge::Exception& e)
	{
		//if there is an error during conversion, display it
		ROS_ERROR("camera::main.cpp::cv_bridge exception: %s", e.what());
		return;
	}
	leftImageExists = true;
	if (rightImageExists)
		if (left_image_ptr->header.stamp.nsec == right_image_ptr->header.stamp.nsec)
		{
			frameCount++;
			if (frameCount > 0)
			{
				frameCount = 0;
				currentTime = left_image_ptr->header.stamp;
				processStereoImages();
			}
		}

}

void rightImageCallback(const sensor_msgs::ImageConstPtr& original_image)
{
	//Convert from the ROS image message to a CvImage suitable for working with OpenCV for processing
	try
	{
		//Always copy, returning a mutable CvImage
		//OpenCV expects color images to use BGR channel order.
		right_image_ptr = cv_bridge::toCvCopy(original_image, enc::MONO8);
	}
	catch (cv_bridge::Exception& e)
	{
		//if there is an error during conversion, display it
		ROS_ERROR("camera::main.cpp::cv_bridge exception: %s", e.what());
		return;
	}
	rightImageExists = true;
	if (leftImageExists)
		if (left_image_ptr->header.stamp.nsec == right_image_ptr->header.stamp.nsec)
		{
			frameCount++;
			if (frameCount > 0)
			{
				frameCount = 0;
				currentTime = left_image_ptr->header.stamp;
				processStereoImages();
			}
		}

}

int main(int argc, char **argv)
{
	/* ros messages */

	/* parameters */
	std::string left_camera_topic;
	std::string right_camera_topic;
	std::string odom_pub_topic;
	std::string point_cloud_pub_topic;
	int loop_rate_param;

	/* initialize ros usage */
	ros::init(argc, argv, "visual_odom");

	/* private nodehandlers */
	ros::NodeHandle nh;
	ros::NodeHandle n("~");

	/* read parameters from ros parameter server if available otherwise use default values */
	n.param<std::string> ("left_camera_topic", left_camera_topic, "/stereo_camera/left/image_rect");
	n.param<std::string> ("right_camera_topic", right_camera_topic, "/stereo_camera/right/image_rect");
	n.param<std::string> ("odom_pub_topic", odom_pub_topic, "/odom");
	n.param<std::string> ("point_cloud_pub_topic", point_cloud_pub_topic, "/visual_odom/point_cloud");

	//Create an ImageTransport instance, initializing it with our NodeHandle.
	image_transport::ImageTransport it(nh);

//	cv::namedWindow("LeftImage", CV_WINDOW_NORMAL);
//	cv::resizeWindow("LeftImage", 640, 480);

//	cv::namedWindow("RightImage", CV_WINDOW_NORMAL);
//	cv::resizeWindow("RightImage", 640, 480);

	image_transport::Subscriber left_image_subcriber = it.subscribe(left_camera_topic.c_str(), 1, leftImageCallback);
	image_transport::Subscriber right_image_subcriber = it.subscribe(right_camera_topic.c_str(), 1, rightImageCallback);

	odom_pub = n.advertise<nav_msgs::Odometry>(odom_pub_topic, 50);
	point_cloud_pub = n.advertise<sensor_msgs::PointCloud>(point_cloud_pub_topic, 10);

	// instantiate transform
	oldTransform = Mat::eye(4,4,CV_64F);

	// instantiate lastTime (used for speed calculations)
	lastTime = ros::Time::now();
	currentTime = ros::Time::now();
	frameCount = 0;
	errorCount = 0;

	// Read camera info
	std::string leftPath = ros::package::getPath("VisualOdom") + "/camera_calibration/left.yaml";
	FileStorage fsLeft(leftPath, FileStorage::READ);
	Mat intrinsicsLeft, distortionLeft, rectificationLeft;
	fsLeft["camera_matrix"] >> intrinsicsLeft;
	fsLeft["distortion_coefficients"] >> distortionLeft;
	fsLeft["rectification_matrix"] >> rectificationLeft;
	fsLeft["projection_matrix"] >> projectionLeft;
	std::string rightPath = ros::package::getPath("VisualOdom") + "/camera_calibration/right.yaml";
	FileStorage fsRight(rightPath, FileStorage::READ);
	Mat intrinsicsRight, distortionRight, rectificationRight;
	fsRight["camera_matrix"] >> intrinsicsRight;
	fsRight["distortion_coefficients"] >> distortionRight;
	fsRight["rectification_matrix"] >> rectificationRight;
	fsRight["projection_matrix"] >> projectionRight;

	if (projectionLeft.cols == 0 || projectionRight.cols == 0) {
		ROS_ERROR("Failed to load camera calibration - Node will shutdown");
	}
	else {
		ROS_INFO("Camera calibration loaded succesfully");
		ros::spin();
	}


	//cv::destroyWindow("LeftImage");
	//cv::destroyWindow("RightImage");
	
}
