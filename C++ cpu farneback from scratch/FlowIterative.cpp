#include "Header.h"


void flow_Iterative(Mat &f1, Mat &f2, Mat &c1, Mat &c2,float sigma ,float sigma_flow, Mat &dx, Mat &dy,bool initial) {
	int num_iter = 3;
	int rows = f1.rows;
	int cols = f1.cols;



	vector<vector<Eigen::MatrixXf>>A1(2, vector<Eigen::MatrixXf>(2, Eigen::MatrixXf(f1.rows, f1.cols)));
	vector<vector<Eigen::MatrixXf>>A2(2, vector<Eigen::MatrixXf>(2, Eigen::MatrixXf(f1.rows, f1.cols)));
	vector<Eigen::MatrixXf>B1(2, Eigen::MatrixXf(f1.rows, f1.cols));
	vector<Eigen::MatrixXf>B2(2, Eigen::MatrixXf(f1.rows, f1.cols));
	Eigen::MatrixXf C1(f1.rows, f1.cols);
	Eigen::MatrixXf C2(f1.rows, f1.cols);







	polynomialExpansionCoeff(f1, c1, sigma, A1, B1, C1);
	polynomialExpansionCoeff(f2, c2, sigma, A2, B2, C2);
	cout << "polynomialExpansionCoeff" << endl;

	/*cout << B1[0] << endl;
	cout << B1[1] << endl;

	cout << B2[0] << endl;
	cout << B2[1] << endl;*/


	Eigen::MatrixXi x_coordinates(rows, cols);
	Eigen::MatrixXi y_coordinates(rows,cols);
	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			x_coordinates(row,col) = col;
			
		}
	}

	for (int col= 0; col < cols; col++) {
		for (int row = 0; row < rows; row++) {
			y_coordinates(row,col) = row;
		}

	}


	/*if (initial) {
		dx= Mat (rows,cols, CV_32F);
		dy= Mat(rows, cols, CV_32F);
	}*/


	int n_flow;
	n_flow =int( 4 * sigma_flow + 1);

	vector<float>w (int(n_flow*2+1));

	for (int i = -n_flow; i <= n_flow; i++)
	{
		w[i]=exp(-(pow(float(i) , 2) / (2*pow(sigma_flow,2))));

	}




	Eigen::MatrixXf S(2,2);
	S(0,0) = 1.0;
	S(1,1) = 1.0;

	Eigen::MatrixXi dx_(rows, cols);
	Eigen::MatrixXi dy_(rows, cols);
	Eigen::MatrixXi x_coordinates_(rows, cols);
	Eigen::MatrixXi y_coordinates_(rows, cols);

	vector<Eigen::MatrixXf>delB(B1.size(), Eigen::MatrixXf(rows, cols));
	vector<vector<Eigen::MatrixXf>>A(2, vector<Eigen::MatrixXf>(2, Eigen::MatrixXf(f1.rows, f1.cols)));
	Eigen::MatrixXf m(2, 2);
	Eigen::VectorXf b(2);
	Eigen::VectorXf u(2);
	Eigen::MatrixXf a_t(2, 2);
	Eigen::MatrixXf a(2, 2);
	Eigen::VectorXf delb(2);
	Eigen::MatrixXf ata(2, 2);
	Eigen::VectorXf atb(2);
	Eigen::MatrixXf G(2, 2);
	Eigen::VectorXf H(2);
	vector<float>result(2);
	vector<float>result2(1);

	vector<vector<Eigen::MatrixXf>>ATA(2, vector<Eigen::MatrixXf>(2, Eigen::MatrixXf(rows, cols)));
	vector<Eigen::MatrixXf>ATB(2, Eigen::MatrixXf(rows, cols));
	vector<vector<Eigen::MatrixXf>>A_T(2, vector<Eigen::MatrixXf>(2, Eigen::MatrixXf(rows, cols)));

	cout << "start iter" << endl;
	for (int itr = 0; itr < num_iter; itr++) {
		cout << "iter= " << itr<<endl;
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				dx_(row, col) = int(dx.at<float>(row, col));
				dy_(row, col) = int(dy.at<float>(row, col));
				x_coordinates_(row,col) = check_bounds(x_coordinates(row,col) +dx_(row,col),cols-1);
				y_coordinates_(row, col) = check_bounds(y_coordinates(row, col) + dy_(row, col), rows-1);

				if ((x_coordinates_(row,col) !=(x_coordinates(row,col) + dx_(row,col))) || (y_coordinates_(row, col) != (y_coordinates(row, col) + dy_(row, col)))) {
					c1.at<float>(row, col) = 0.0;
				}

				for (int i = 0; i < 2; i++) {
					for (int j = 0; j <2; j++) {
						A[i][j](row,col) = c1.at<float>(row, col) *(A1[i][j](row, col) + A2[i][j](y_coordinates_(row,col),x_coordinates_(row,col)) / 2.0);
					}
				}
			}
		}
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				b(0) = dx_(row,col);
				b(1) = dy_(row,col);

				m(0,0) = A[0][0](row,col);
				m(0, 1) = A[0][1](row, col);
				m(1, 0) = A[1][0](row, col);
				m(1, 1) = A[1][1](row, col);


				u = m * b;
				//cout << m << endl << b << endl << u << endl;

				for (int i = 0; i < B1.size(); i++) {
					delB[i](row,col) = c1.at<float>(row,col)*(-1 / 2) * (B2[i](y_coordinates_(row,col),x_coordinates_(row,col)) - B1[i](row,col))+u(i);	
					//cout << c1.at<float>(row, col) << " " << B2[i](y_coordinates_(row, col), x_coordinates_(row, col)) - B1[i](row, col) << " "<<u(i) << endl;
				}
			}
		}

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				A_T[0][0](row, col) = A[0][0](row, col);
				A_T[0][1](row, col) = A[1][0](row, col);
				A_T[1][0](row, col) = A[0][1](row, col);
				A_T[1][1](row, col) = A[1][1](row, col);
			}
		}



		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				a_t(0, 0) = A_T[0][0](row, col);
				a_t(0, 1) = A_T[0][1](row, col);
				a_t(1, 0) = A_T[1][0](row, col);
				a_t(1, 1) = A_T[1][1](row, col);


				a(0, 0) = A[0][0](row, col);
				a(0, 1) = A[0][1](row, col);
				a(1, 0) = A[1][0](row, col);
				a(1, 1) = A[1][1](row, col);



				delb(0) = delB[0](row,col);
				delb(1) = delB[1](row, col);


				

				ata = S* a_t * a * S;

				ATA[0][0](row, col) = ata(0, 0);
				ATA[0][1](row, col) = ata(0, 1);
				ATA[1][0](row, col) = ata(1, 0);
				ATA[1][1](row, col) = ata(1, 1);



				atb = S * a_t * delb;


				ATB[0](row, col) = atb(0);
				ATB[1](row, col) = atb(1);

			
			
			}
		}

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {



				//axis=0
				correlate1d(vector<float>{ ATA[0][0](row, col),ATA[0][1](row, col)}, w, result);
				G(0, 0) = result[0];
				G(0, 1) = result[1];

				correlate1d(vector<float>{ ATA[1][0](row, col), ATA[1][1](row, col)}, w, result);
				G(1, 0) = result[0];
				G(1, 1) = result[1];

				//axis=1
				correlate1d(vector<float>{G(0, 0), G(1, 0)}, w, result);
				G(0, 0) = result[0];
				G(1, 0) = result[1];
				correlate1d(vector<float>{G(0, 1), G(1, 1)}, w, result);
				G(0, 1) = result[0];
				G(1, 1) = result[1];

				//axis=0
				correlate1d(vector<float>{ATB[0](row,col), ATB[1](row, col)}, w, result);

				H(0) = result[0];
				H(1) = result[1];

				//axis=1

				correlate1d(vector<float>{H(0)}, w, result2);
				H(0) = result2[0];
				correlate1d(vector<float>{H(1)}, w, result2);
				H(1) = result2[0];

				u = G.colPivHouseholderQr().solve(H);
				u = S * u;


				dx.at<float>(row, col) = u(0);
				dy.at<float>(row, col) = u(1);
				//cout << G << " " << H << " " << u << endl;
			}
		}
		

	}

	//for (int i = 0; i < 2;i++) {
	//	for (int j = 0; j < 2; j++) {
	//		cout << A[i][j] << endl;
	//		cout << ATA[i][j] << endl;
	//		cout << A_T[i][j] << endl;


	//	}
	//	cout << delB[i] << endl;
	//	cout << ATB[i] << endl;

	//}

	cout << "flowiterative" << endl;
	cout << dx << endl;

}