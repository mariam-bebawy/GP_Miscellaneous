Point getFinalPosition(GpuMat vel_norm, GpuMat corr_norm, GpuMat im_norm, int w_corr, int w_im, int w_final) {
    int rows = vel_norm.rows;
    int cols = vel_norm.cols;
    int n = rows * cols;
    GpuMat C_corr(rows, cols);
    GpuMat C_im(rows, cols);
    GpuMat comb(rows, cols);
    GpuMat final(rows, cols);
    GpuMat xpos(rows, cols);
    GpuMat ypos(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float velVal = vel_norm.get(i, j);
            float CcorrVal = velVal * (w_corr * corr_norm.get(i, j));
            float CimVal = velVal * (w_im * im_norm.get(i, j));
            float combVal = CcorrVal + CimVal;
            float finalVal = pow(combVal, w_final);

            C_corr.set(i, j, CcorrVal);
            C_im.set(i, j, CimVal);
            comb.set(i, j, combVal);
            final.set(i, j, finalVal);

            xpos.set(i, j, (finalVal * i));
            ypos.set(i, j, (finalVal * j));
        }
    }

    Scalar sum;
    sum = sum(final);
    float sumD = sum[0];
    sum = sum(xpos);
    float sumN_x = sum[0]
    sum = sum(ypos);
    float sumN_y = sum[0];

    int posx = sumN_x / sumD;
    int posy = sumN_y / sumD;

    Point pos(posx, posy);

    return pos;
}