"""
$ v4l2-ctl --list-formats-ext
ioctl: VIDIOC_ENUM_FMT
	Index       : 0
	Type        : Video Capture
	Pixel Format: 'RG10'
	Name        : 10-bit Bayer RGRG/GBGB
		Size: Discrete 3280x2464 / 0.048s (21.000 fps)
		Size: Discrete 3280x1848 / 0.036s (28.000 fps)
		Size: Discrete 1920x1080 / 0.033s (30.000 fps)
		Size: Discrete 1280x720  / 0.017s (60.000 fps)
		Size: Discrete 1280x720  / 0.017s (60.000 fps)
"""
# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen
# Pixel Count: 3280 x 2464 (active pixels) 3296 x 2512 (total pixels)

import cv2

def gstreamer_pipeline(
        sensor_id=0,
        sensor_mode=3,
        capture_width=3296,
        capture_height=2512,
        display_width=1280,
        display_height=720,
        framerate=60,
        flip_method=0,
    ) -> str:
    """ get_commends string"""
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    rate = 0.4
    rate_width = int(1280 * rate)   # 3280
    rate_height = int(720 * rate)   # 2464

    cap = cv2.VideoCapture(
            gstreamer_pipeline(
                sensor_id=0,
                flip_method=0,
                display_width=rate_width,
                display_height=rate_height,
                ), 
            cv2.CAP_GSTREAMER
        )

    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
    
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)

            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF

            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    show_camera()
