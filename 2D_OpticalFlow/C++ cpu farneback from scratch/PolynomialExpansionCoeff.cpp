#include "Header.h"

void polynomialExpansionCoeff(Mat &f, Mat &c, float sigma, vector<vector<Eigen::MatrixXf>>&A, vector<Eigen::MatrixXf>&B, Eigen::MatrixXf &C){


	int n = int(4 * sigma + 1);
	Eigen::VectorXf x((2 * n) + 1);
	Eigen::VectorXf a((2 * n) + 1);

	for (int i = 0; i <= 2 * n+1; i += 1) {
		int xval = - n + i;
		x[i] = xval;
		a[i] = (exp(-(xval * xval) / (2 * sigma * sigma)));
	}


	Eigen::MatrixXf bx(a.size(),6);
	for (int i = 0; i < a.size(); i++) {
		bx(i, 0) = 1;
		bx(i, 1) = x[i];
		bx(i, 2) = 1;
		bx(i, 3) = pow(x[i], 2);
		bx(i, 4) = 1;
		bx(i, 5) = x[i];
	}

	Eigen::MatrixXf by(a.size(), 6);
	for (int i = 0; i < a.size(); i++) {
		by(i, 0) = 1;
		by(i, 1) = 1;
		by(i, 2) = x[i];
		by(i, 3) = 1;
		by(i, 4) = pow(x[i], 2);
		by(i, 5) = x[i];
	}

	Mat cf(f.rows,f.cols, CV_32F);

	for (int row = 0; row < f.rows; row++) {
		for (int col = 0; col < f.cols; col++) {
			cf.at<float>(row, col) = f.at<float>(row, col) * c.at<float>(row, col);
		}
	}


	Eigen::MatrixXf ab(a.size(),6);
	for (int i = 0; i < a.size(); ++i) {
		for (int j = 0; j <6; ++j) {
			ab(i,j) = a[i] * bx(i,j);
		}
	}


	vector<Eigen::MatrixXf>abb(a.size(), Eigen::MatrixXf( 6,6));
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < 6; j++) {
			for (int k = 0; k < 6; k++) {
				abb[i](j,k) = ab(i,j) * bx(i,k);
			}
		}
	}

	vector<vector<Eigen::MatrixXf>>G(6, vector<Eigen::MatrixXf>(6, Eigen::MatrixXf(f.rows, f.cols)));
	vector<Eigen::MatrixXf>v(6, Eigen::MatrixXf(f.rows, f.cols));
	

	vector<float>abb_ij(abb.size(),0);
	vector<float>ab_i(abb.size(), 0);
	vector<float>temp_col(f.rows);
	vector<float>temp_col_result(f.rows);
	vector<float>temp_row(f.cols);
	vector<float>temp_row_result(f.cols);
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j <6; j++) {
			for (int k = 0; k < abb.size(); k++) {
					abb_ij[k]=abb[k](i,j);
			}
			
			for (int row = 0; row < f.rows; row++) {
				
				c.row(row).copyTo(temp_row);
				correlate1d(temp_row,abb_ij, temp_row_result);
				for (int col = 0; col < f.cols; col++) {
					G[i][j](row, col) = temp_row_result[col];

				}
			}
		}
		for (int k = 0; k < ab.size(); k++) {
			ab_i[k] = ab(k, i);
		}
		for (int row = 0; row < f.rows; row++) {
			cf.row(row).copyTo(temp_row);

			correlate1d(temp_row, ab_i, temp_row_result);
			for (int col = 0; col < f.cols; col++) {
				v[i](row, col) = temp_row_result[col];


			}
		}
	}
	cout << "correlate1d x" << endl;
	//cout << v[0] << endl << v[1] << endl << v[2] << endl << v[3] << endl << v[4] << endl << v[5] << endl;

	for (int i = 0; i < a.size(); ++i) {
		for (int j = 0; j < 6; ++j) {
			ab(i, j) = a[i] * by(i, j);
		}
	}

	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < 6; j++) {
			for (int k = 0; k < 6; k++) {
				abb[i](j, k) = ab(i, j) * by(i, k);
			}
		}
	}

	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			for (int k = 0; k < abb.size(); k++) {
				abb_ij[k] = abb[k](i, j);
			}

			for (int col = 0; col < f.cols; col++) {
				for (int row = 0; row < f.rows; row++) {
					temp_col[row]=(G[i][j](row, col));
				}
				correlate1d(temp_col, abb_ij, temp_col_result);
				for (int row = 0; row < f.rows; row++) {
					G[i][j](row, col) = temp_col_result[row];
				}
			}
		}
		for (int k = 0; k < ab.size(); k++) {
			ab_i[k] = ab(k,i);
		}
		for (int col = 0; col < f.cols; col++) {
			for (int row = 0; row < f.rows; row++) {
				temp_col[row]=(v[i](row, col));
			}
 			correlate1d(temp_col, ab_i, temp_col_result);
			for (int row = 0; row < f.rows; row++) {
				v[i](row, col) = temp_col_result[row];
			}
		}
	}

	cout << "correlate1d y" << endl;
	//cout << v[0] << endl << v[1] << endl << v[2] << endl << v[3] << endl << v[4] << endl << v[5] << endl;


	vector<Eigen::MatrixXf > r(6, Eigen::MatrixXf(f.rows,f.cols));

	
	Eigen::MatrixXf m(6,6);
	Eigen::VectorXf b(6);
	Eigen::VectorXf u(6);
	
	for (int row = 0; row < f.rows; row++) {
		for (int col = 0; col < f.cols; col++) {
			for (int i = 0; i <6; i++) {
				for (int j = 0; j < 6; j++) {
					m(i,j) = G[i][j](row,col);
				}
				b(i) = v[i](row,col);

			}
			u= m.colPivHouseholderQr().solve(b);
			//cout << m << endl << b << endl << u<<endl;
			for (int i = 0; i < 6; i++) {
				r[i](row,col)= u(i);
			}
		}
	}
	cout << "almost there" << endl;

	A[0][0] = r[3];
	A[0][1] = r[5] / 2.0;
	A[1][0] = A[0][1];
	A[1][1] = r[4];
	B[0]=r[1];
	B[1]=r[2];
	C = r[0];

	/*for (int i = 0; i < 6;i++) {
		for (int j = 0; j < 6; j++) {
			cout << G[i][j] << endl;
		}
		cout << v[i] << endl;
		cout << r[i] << endl;
	}*/

}


