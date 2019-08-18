#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/CameraInfo.h"
#include "opencv2/imgproc/detail/distortion_model.hpp"
#include<string>
#include<algorithm>
#define _NODE_NAME_ "camera_driver_node"

using namespace cv;
using namespace std;

class ImageTalker
{
public:
	ImageTalker()
	{
	}
	bool init()
	{
		ros::NodeHandle nh,nh_private("~");
		
		image_transport::ImageTransport it(nh);
	
		std::string calibrationFilePath;
		
		if(!ros::param::get("~calibration_file_path",calibrationFilePath))
		{
			ROS_ERROR("[%s]: please set calibration_file_path parameter",_NODE_NAME_);
			return false;
		}
		
		nh_private.param<bool>("is_show_image",m_is_show,false);
		
		nh_private.param<int>("frame_rate",m_frameRate,30);
		
		if(!ros::param::get("~cameras_id",m_camerasId))
		{
			ROS_ERROR("[%s]: please set cameras_id parameter",_NODE_NAME_);
			return false;
		}
		
		m_distortCoefficients.resize(m_camerasId.size());
		m_instrinsics.resize(m_camerasId.size());
		m_newInstrinsics.resize(m_camerasId.size());
		
		for(int i=0; i<m_camerasId.size(); ++i)
		{
			std::string file_name = calibrationFilePath + std::to_string(i+1) + ".yaml";
			if(!loadIntrinsics(file_name,m_instrinsics[i],m_distortCoefficients[i]))
				return false;
			//m_newInstrinsics[i] = getOptimalNewCameraMatrix(m_instrinsics[i],m_distortCoefficients[i],m_imgSize,1.0);
		}
		
		m_pub = it.advertise("/image_rectified", 1);
		
		//camera_info_pub_ = nh.advertise<sensor_msgs::CameraInfo>("/camera_info", 1);
		//timer_ = nh.createTimer(ros::Duration(0.2), &ImageTalker::timerCallback, this);
	}
	
	void timerCallback(const ros::TimerEvent& event)
	{
	}
	
	void run()
	{
		std::vector<cv::VideoCapture> cameraHandles(m_camerasId.size());
		for(int i=0; i<m_camerasId.size();++i)
		{
			if(!cameraHandles[i].open(m_camerasId[i]))
			{
				ROS_ERROR("[%s] open camera %d failed",_NODE_NAME_,m_camerasId[i]);
				return;
			}
			cameraHandles[i].set(CV_CAP_PROP_FPS,m_frameRate);
			cameraHandles[i].set(CV_CAP_PROP_FRAME_WIDTH,m_imgSize.width);
			cameraHandles[i].set(CV_CAP_PROP_FRAME_HEIGHT,m_imgSize.height);
		}
		ros::Rate loop_rate(m_frameRate);
		std::vector<cv::Mat> raw_images(m_camerasId.size());
		std::vector<cv::Mat> rectified_images(m_camerasId.size());
		sensor_msgs::Image::Ptr rosImage;
		cv::Mat resultImage(m_imgSize.height,m_imgSize.width*m_camerasId.size(),CV_8UC3);
	
		while(ros::ok())
		{
			for(int i=0; i<cameraHandles.size(); ++i)
				cameraHandles[i].grab();
			for(int i=0; i<cameraHandles.size(); ++i)
			{
				cameraHandles[i].retrieve(raw_images[i]);
				//cv::flip(src,dst,-1); //turn image
				cv::undistort(raw_images[i], rectified_images[i], m_instrinsics[i],
							 m_distortCoefficients[i],m_newInstrinsics[i]);
				rectified_images[i].copyTo(resultImage(
					Rect(rectified_images[i].cols*(cameraHandles.size()-1-i),0,m_imgSize.width,m_imgSize.height)));
					
				//imshow(std::to_string(i+1),rectified_images[i]);
				
			}
				namedWindow("resultImage",WINDOW_NORMAL);
				imshow("resultImage",resultImage);
				waitKey(1);
			loop_rate.sleep();
		}
	}
		
		


//cv::namedWindow("image_raw",cv::WINDOW_NORMAL); 
//cv::imshow("image_raw",frame);
//msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
//msg->header.frame_id="camera";
//pub_.publish(msg);

	
	bool loadIntrinsics(const std::string &calibration_file,
						cv::Mat &camera_instrinsic,
						cv::Mat &distortion_coefficients)
	{
		if (calibration_file.empty())
		{
			ROS_ERROR("[%s] missing calibration file path", _NODE_NAME_);
			return false;
		}

		cv::FileStorage fs(calibration_file, cv::FileStorage::READ);

		if (!fs.isOpened())
		{
			ROS_INFO("[%s] cannot open calibration file %s", _NODE_NAME_, calibration_file.c_str());
	 		return false;
		}

		camera_instrinsic = cv::Mat(3, 3, CV_64F);
		distortion_coefficients = cv::Mat(1, 5, CV_64F);

		cv::Mat dis_tmp;
		
		fs["CameraMat"] >> camera_instrinsic;
		fs["DistCoeff"] >> dis_tmp;
		fs["ImageSize"] >> m_imgSize;
		fs["DistModel"] >> m_distModel;
		
		for (int col = 0; col < 5; col++)
			distortion_coefficients.at<double>(col) = dis_tmp.at<double>(col);
		fs.release();	//释放
		return true;
	}


private:
	bool m_is_rectify;
	image_transport::Publisher m_pub;
	ros::Timer m_timer;
	
	std::vector<int> m_camerasId;
	std::vector<cv::Mat> m_newInstrinsics;
	std::vector<cv::Mat> m_instrinsics;
	std::vector<cv::Mat> m_distortCoefficients;
	std::string m_distModel;
	cv::Size m_imgSize;
	int m_frameRate;
	bool m_is_show;
	
};


int main(int argc, char** argv)
{
	ros::init(argc, argv, _NODE_NAME_);
	
	ImageTalker image_talker;
	if(image_talker.init())
		image_talker.run();
	return 0;
}
