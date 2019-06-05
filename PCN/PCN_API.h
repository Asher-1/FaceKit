#ifndef __PCN_API__
#define __PCN_API__
#include "PCN.h"

#define kFeaturePoints 14

extern "C"
{
	void *init_detector(const char *detection_model_path, 
			const char *pcn1_proto, const char *pcn2_proto, const char *pcn3_proto, 
			const char *tracking_model_path, const char *tracking_proto,
			int min_face_size, float pyramid_scale_factor, float detection_thresh_stage1,
			float detection_thresh_stage2, float detection_thresh_stage3, int tracking_period,
			float tracking_thresh)
	{
		PCN *detector = new PCN(detection_model_path,pcn1_proto,pcn2_proto,pcn3_proto,
				tracking_model_path,tracking_proto);

		/// detection
		detector->SetMinFaceSize(min_face_size);
		detector->SetImagePyramidScaleFactor(pyramid_scale_factor);
		detector->SetDetectionThresh(
				detection_thresh_stage1,
				detection_thresh_stage2,
				detection_thresh_stage3);
		/// tracking
		detector->SetTrackingPeriod(tracking_period);
		detector->SetTrackingThresh(tracking_thresh);
		return static_cast<void*> (detector);
	}
	
	int get_detect_status(void* pcn){
		PCN* detector = (PCN*) pcn;
		return  detector->detectFlag;
	}

	Window* detect_faces(void* pcn, unsigned char* raw_img,size_t rows, size_t cols, int *lwin)
	{
		PCN* detector = (PCN*) pcn;
		cv::Mat img(rows,cols, CV_8UC3, (void*)raw_img);
		std::vector<Window> faces = detector->DetectTrack(img);

		*lwin = faces.size();
		Window* wins = (Window*)malloc(sizeof(Window)*(*lwin));
		memcpy(wins,&faces[0],*lwin * sizeof(Window));
		//for (int i=0; i < *lwin; i++){
		//	wins[i] = faces[i];
		//}
		return wins;
	}

	void free_faces(Window* wins)
	{
		free(wins);
	}

	void free_detector(void *pcn)
	{
		PCN* detector = (PCN*) pcn;
		delete detector;
	}
}



#endif

