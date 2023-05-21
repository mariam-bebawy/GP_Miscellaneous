***steps simply :***  
preprocessing  
initial template matching f awl frame  
optical frame -> da eli by7rk el template  
template matching refinement -> y7sn accuracy -> da el weighting maps  
da byzbt variations el quality byst5dm el NCC  
boundary detection using graphcut algorithm  
  
***pseudo code :***  
```python
// Define parameters
stemplate = size of the template
NeighborhoodSize = 5
PyramidScale = 0.5
NumPyramidLevels = 3
FilterSize = 5
MaxVelocityMagnitude = 5
MaxPositionDifference = 20

// Initialize variables
position = initial position
velocity = 0
regionSize = stemplate

while true:
    // Reduce image to region of interest
    roi = reduceToROI(image, position, regionSize)

    // Calculate optical flow
    flow = calculateOpticalFlow(roi, NeighborhoodSize, PyramidScale, NumPyramidLevels, FilterSize)

    // Estimate velocity
    velocity = calculateAverageVelocity(flow, position, stemplate)

    // Adjust moving region if necessary
    if isCloseToEdge(position, regionSize):
        regionSize = adjustRegionSize(position, regionSize)

    // Generate weighting map
    weightingMap = generateWeightingMap(position, velocity)

    // Perform template matching to get peak correlation position
    correlationPosition = templateMatching(roi, weightingMap)

    // Check for large velocity and position difference
    if magnitude(velocity) > MaxVelocityMagnitude and distance(position, correlationPosition) > MaxPositionDifference:
        velocity = 0

    // Update position
    position = position + velocity

    // Check termination condition
    if terminationConditionMet():
        break

// Final position estimation
estimatedPosition = position
```