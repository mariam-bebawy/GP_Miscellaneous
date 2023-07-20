/*

GpuMat normalizeMaps(GpuMat map) {
    int sizeMap = map.size();
    double minMap = map.min();
    double maxMap = map.max();

    GpuMat norm{sizeMap};
    
    for( i=0; i<sizeMap; i++) {
        norm[i] = ( map[i] - minMap )/( maxMap - minMap );
    }

    return norm;
}

*/

GpuMat normalizeMaps(GpuMat map) {
    float minVal, maxVal;
    map.MinMax(minVal, maxVal);

    int rows = map.rows;
    int cols = map.cols;
    GpuMat mapNorm(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float val = map.get(i, j);
            val = (val - minVal) / (maxVal - minVal);
            mapNorm.set(i, j, val);
        }
    }

    return mapNorm;
}
