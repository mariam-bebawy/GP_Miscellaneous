void showResult(Mat frame, Point pos, int radius = 100) {
    // calc box corners' coordinates
    Point boxTopLeft(pos.x - radius, pos.y - radius);
    Point boxBottomRight(pos.x + radius, pos.y + radius);

    rectangle(frame, boxTopLeft, boxBottomRight, Scalar(0, 0, 255), 2);
    circle(frame, pos, radius/4, Scalar(255, 0, 0), 2);

    imshow("result", frame);
    waitKey(0);
}