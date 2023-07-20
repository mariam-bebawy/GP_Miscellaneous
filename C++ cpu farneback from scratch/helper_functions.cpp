#include "Header.h"
void printVec(vector<vector<double>> vec) {
    for (int i = 0; i < vec.size(); i++) {
        for (int j = 0; j < vec[i].size(); j++) {
            cout << vec[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;


}
void create_CertaintyMatrix(int rows, int cols, Mat &c) {

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            c.at<float>(Point(j,i)) = float(min(min(1.0, 0.2 * min(i, j)), 0.2 * min(rows - 1 - i, cols - 1 - j)));
        }
    }
}

vector<vector<double>> reduceImageSize(vector<vector<double>> image, int scalefactor) {
    if (scalefactor <= 0) {
        return image;
    }
    vector<vector<double>>scaledImage(int(image.size() / scalefactor) + 1, vector<double>(int(image[0].size() / scalefactor) + 1, 0));
    for (int r = 0; r < image.size(); r = r + scalefactor) {
        for (int c = 0; c < image[0].size(); c = c + scalefactor) {
            scaledImage[int(r / scalefactor)][int(c / scalefactor)] = image[r][c];
        }
    }
    return scaledImage;
}

vector<vector<double>> expandImage(vector<vector<double>> image) {
    vector<vector<double>>expandedImage(int(image.size() * 2), vector<double>(int(image[0].size() * 2), 0));
    // Iterate over the original image and copy it over to the expanded image
    for (int i = 0; i < image.size(); i++) {
        for (int j = 0; j < image[0].size(); j++) {
            expandedImage[2 * i][2 * j] = image[i][j];
            expandedImage[2 * i][2 * j + 1] = image[i][j];
            expandedImage[2 * i + 1][2 * j] = image[i][j];
            expandedImage[2 * i + 1][2 * j + 1] = image[i][j];
        }
    }

    // Return the new, expanded image
    return expandedImage;
}

int check_bounds(int x, int max) {
    if (x >= max)
        return max - 1;
    if (x < 0)
        return 0;
    return x;
}

vector<double> Vector_Matrix_Multiplication(vector<vector<double>> V1, vector<double>V2) {
    // v1 2*2
    // v2 2*1
    vector<double>result(V1.size());
    result[0] = (V1[0][0] * V2[0]) + (V1[0][1] * V2[1]);
    result[1] = (V1[1][0] * V2[0]) + (V1[1][1] * V2[1]);


    return result;
}

vector < vector<double>> Matrix_Multiplication(vector<vector<double>> V1, vector<vector<double>> V2) {
    // v1 2*2
    // v2 2*2
    vector<vector<double>> result(V1.size(), vector<double>(V2[0].size(), 0));
    for (int i = 0; i < V1.size(); i++)
    {
        for (int j = 0; j < V1.size(); j++)
        {
            result[i][j] = 0;

            for (int k = 0; k < V1.size(); k++)
            {
                result[i][j] += V1[i][k] * V2[k][j];
            }
        }
    }

    return result;
}

std::vector< std::vector<double> > cvMat_2D2vec(Mat mat) {
    int rows = static_cast<int>(mat.rows);
    int cols = static_cast<int>(mat.cols);
    std::vector< std::vector<double> >vec(rows, std::vector<double>(cols, 0));
    for (int i = 0; i < rows; ++i) {
        mat.row(i).copyTo(vec[i]);
    }
    return vec;
}
void correlate1d(vector<float> input, vector<float> filter, vector<float>& result)
{

    // define the output vector
    vector<float> new_input(input.size() + filter.size() -1, 0);

    for (int i = 0; i < input.size(); i++) {
        new_input[i + (filter.size()/2)] = input[i];


    }
    //printVec(new_input);
    // loop through the input vector
    for (int i = 0; i < input.size(); i++) {
        int sum = 0;
        //cout<<i<<endl;
        for (int j = 0; j < filter.size(); j++)
        {
            sum += new_input[j + i] * filter[j];
        }

        // add the sum to the output vector
        result[i] = sum;
    }
}

// Function to swap the axes of a 2D vector
vector<vector<double>> swapAxes(vector<vector<double>> vect)
{
    vector<vector<double>> result(vect[0].size(), vector<double>(vect.size(), 0));
    // Swap the rows and columns'
    for (int i = 0; i < vect.size(); i++)
    {
        for (int j = 0; j < vect[i].size(); j++)
        {
            // Swap the elements 
            result[j][i] = vect[i][j];
        }
    }
    return result;
}

vector<double> solveMatrix(vector<vector<double>> matrix1, vector<double> matrix2) {
    int n = matrix1.size(); // number of rows
    int m = matrix2.size(); // number of columns
    // augment the matrices
    vector<vector<double>> augmentedMatrix(n, vector<double>(m + 1));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            augmentedMatrix[i][j] = matrix1[i][j];
        }
        augmentedMatrix[i][m] = matrix2[i];
    }
    //row reduce
    for (int i = 0; i < n; i++) {
        double pivot = augmentedMatrix[i][i];
        for (int j = i; j <= m; j++) {
            augmentedMatrix[i][j] /= pivot;
        }
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double multiplier = augmentedMatrix[k][i];
                for (int j = i; j <= m; j++) {
                    augmentedMatrix[k][j] -= multiplier * augmentedMatrix[i][j];
                }
            }
        }
    }
    //get solution
    vector<double> solution(m);
    for (int i = 0; i < m; i++) {
        solution[i] = augmentedMatrix[i][m];
    }
    return solution;
}
template <typename T>
cv::Mat_<T> vec2cvMat_2D(std::vector< std::vector<T> >& inVec) {
    int rows = static_cast<int>(inVec.size());
    int cols = static_cast<int>(inVec[0].size());

    cv::Mat_<T> resmat(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        resmat.row(i) = cv::Mat(inVec[i]).t();
    }
    return resmat;
}
