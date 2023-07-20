#include "functions.h"



vector<vector<Mat>> integrateflowfield(vector<Mat>* flowfields,Mat Mask_nuc) {


	//Mask_nuc.convertTo(Mask_nuc, CV_32F);
	Mat flow_parts[2];
	vector<Mat> u (flowfields->size());
	vector<Mat> v (flowfields->size());

	for (int i = 0; i < flowfields->size(); i++) {
		split(flowfields->at(i), flow_parts);
		//cv::bitwise_and(flow_parts[0], Mask_nuc, flow_parts[0]);
		//cv::bitwise_and(flow_parts[1], Mask_nuc, flow_parts[1]);
		//flow_parts[0] = flow_parts[0].mul(Mask_nuc);
		//flow_parts[1] = flow_parts[1].mul(Mask_nuc);
		flow_parts[0].copyTo(u[i], Mask_nuc);
		flow_parts[1].copyTo(v[i], Mask_nuc);
	}


	int rows = u[0].rows;
	int cols = u[0].cols;

	Mat x(rows, cols, CV_32F);
	Mat y(rows, cols, CV_32F);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			x.row(i).col(j) = j;
		}
	}

	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {
			y.row(j).col(i) = j;
		}
	}




	vector<Mat> xp, yp;
	for (int k = 0; k < u.size() + 1; k++) {
		Mat zerosmatx(rows, cols, CV_32F, 0.0);
		Mat zerosmaty(rows, cols, CV_32F, 0.0);
		xp.push_back(zerosmatx);
		yp.push_back(zerosmaty);
	}
	
	xp[0] = x;
	yp[0] = y;





	for (int t = 0; t < u.size() ; t++) {
		Mat temp_x, temp_y;

		remap(u[t], temp_x ,xp[t], yp[t], INTER_LINEAR);
		xp[t + 1]= xp[t]+ temp_x;


		remap(v[t], temp_y, xp[t], yp[t], INTER_LINEAR);
		yp[t + 1] = yp[t] + temp_y;

		//cv::bitwise_and(xp[t + 1], Mask_nuc, xp[t + 1]);
		//cv::bitwise_and(yp[t + 1], Mask_nuc, yp[t + 1]);
		//xp[t + 1] = xp[t + 1].mul(Mask_nuc);
		//yp[t + 1] = yp[t + 1].mul(Mask_nuc);





	}
	vector<vector<Mat>> integratedflowfields;
	integratedflowfields.push_back(xp);
	integratedflowfields.push_back(yp);
	
	return integratedflowfields;

}
	