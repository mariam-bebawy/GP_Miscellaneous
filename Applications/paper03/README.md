***pseudo code :***  
```python
// Registration and Motion Estimation (RT)
RT_points = select_reference_points() // Select reference points in the reference frame
filtered_RT_points = []
for each point p in RT_points:
    p_prime = track_point(p) // Track the point p in subsequent frames
    if displacement(p, p_prime) < 4 mm:
        filtered_RT_points.append(p_prime)
        
valid_RT = (len(filtered_RT_points) > 0.4 * len(RT_points))

if valid_RT:
    affine_transform = compute_affine_transform(filtered_RT_points)

// Vessel Size and Position Refinement
for each frame fi:
    for each POI p in reference_frame:
        template_matching_result = template_matching(p, fi) // Perform template matching
        if normalized_cross_correlation(template_matching_result) < 0.3:
            // Non-vessel structure tracking
            track_point_with_RT_and_IT(p)
        else:
            // Vessel tracking
            if vessel_size(p) > 10 px:
                vessel_representation = detect_vessel_with_star(fi)
            else:
                vessel_representation = detect_vessel_with_template_matching(fi)
                
            vessel_representation = refine_vessel_representation(vessel_representation, affine_transform)
            constrain_vessel_size(vessel_representation) // Restrict vessel size within [75%..120%] of initial size

// Template-based Reset
reset_threshold = compute_reset_threshold(reference_frame)
for each frame fi:
    reset_candidates = []
    for each POI p in reference_frame:
        reset_candidate = template_matching(p, fi, region_around_initial_position)
        if normalized_cross_correlation(reset_candidate) > reset_threshold:
            reset_candidates.append(reset_candidate)
    update_POI_positions(reset_candidates)

// Motion Tracking Recovery
for each frame fi:
    if valid_RT:
        update_tracked_points_with_Kalman_filter()
    else:
        switch_to_IT_tracking()
    pick_best_position()
    refine_position()
```  
