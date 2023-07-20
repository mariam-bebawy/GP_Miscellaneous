GpuMat calcWeightingMap(int w_alt, vector<GpuMat> alt_temps, GpuMat init_frame, GpuMat curr_frame) {
    int n_alt = alt_temps.size();

    int rows = init_frame.rows;
    int cols = init_frame.cols;
    GpuMat mapCorr(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float x_i = curr_frame.get(i, j);
            float x_init = init_frame.get(i, j);

            // FI SUM HNA MSH 3RFA A3MLO EZAY LSA !!!
            float val = w_alt * x_i + x_init;
            mapCorr.set(i, j, val);
        }
    }

    return mapCorr;
}