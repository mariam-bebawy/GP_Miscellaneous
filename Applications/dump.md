## PAPER02 - *cpp CODE*  
---  
<br>

`GpuMat calcWeightingMap(int w_alt, vector<GpuMat> alt_temps, GpuMat init_frame, GpuMat curr_frame);`  
* The 2D normalized cross-correlation is then computed for both the original template and each of the alternatives and the results weighted and summed. This process creates a weighting map allowing the determination of the location within the image that best matches both the initial template and the templates containing any subsequent morphological changes that may have occurred.  
* $${M_{corr} = \sum_{i=1}^{n_{alt}} \space (w_{alt} \space . \space X_i) \space + \space X_{init}}$$  
* ${M_{corr}}$ the weighting map based on normalized cross-correlation  
* ${n_{alt}}$ the number of alternative templates  
* ${w_{alt}}$ the weighting of the alternatives  
* ${X_i}$ the cross-correlation of the alternative template and current frame for alternative ${i}$  
* ${X_{init}}$ the cross-correlation of the initial template and current frame  

***NOTES IN CODE***  
* weighting map based on normalized cross - correlation, nalt the number of alternative templates, walt the weighting of the alterna
* remember to invert the current frame before M_im according to "intensity at the desired point is lower than the mean intensity throughout the region encompassed by the initial template"
---  
<br>

`GpuMat normalizeMaps(GpuMat weighting_map);`  
* Each of the generated weighting maps are first normalized such that the minimum value in each becomes zero, and the maximum becomes one  
* $${M_{NORM}=\frac{M_{any} - min(M_{any})} {max(M_{any}) - min(M_{any})}}$$  
* the generated position/velocity and template matching weighting maps are normalized to between 0 and 1  

***NOTES IN CODE***  
* normalize Ms  
---  
<br>

`Point getFinalPosition(GpuMat M_final);`  
* the generated position/velocity and template matching weighting maps are normalized, weighted and multiplied together to provide a combined weighting map  
* $${M_{Ccorr} = M_{Nvel} . (w_{corr} . M_{Ncorr})}$$  
*  the generated image intensity map is normalized and combined with the position velocity weighting map  
* $${M_{Cim} = M_{Nvel} . (w_{im} . M_{Nim})}$$  
*  the two combined maps are then summed  
* $${M_{comb} = M_{Ccorr} + M_{Cim}}$$  
*  *special case:* if the point is close to the edge of the image such that there is an overlap between the edge and template, only the image-based weighting map is used
* $${M_{comb} = M_{Cim}}$$  
* the resulting values are then raised to a further final weighting-power  
* $${M_{final} = M_{comb}\space^{w_{final}}}$$  
* the position of the object within the frame is then calculated as the weighted mean position, with the weight values taken from this final map  
* $${x_k = \frac {\sum_{i=1}^n (M_{final}^i\space.\space x_i)} {\sum_{i=1}^n (M_{final}^i)}}$$  
* ${x_k}$ the predicted position of the object in frame ${k}$  
* ${n}$ the number of pixels in the final weighting map  
* ${M_{final}^i}$ the value of the final weighting map for pixel number ${i}$  
* ${x_i}$ is the position of pixel number ${i}$  
*  resulting point is then utilized to initialize the velocity-based weighting map of the following frame  

***NOTES IN CODE***  
* get the predicted position knowing normalizaed M_corr, M_vel , M_im  
---  
<br>

`void showResult(Mat frame);`  
* draw bounding box around point with the size of template  
* draw circle around point  

***NOTES IN CODE***  
* plot point  