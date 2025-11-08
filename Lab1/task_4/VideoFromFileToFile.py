import cv2

def readIPWriteTOFile():
    videoThread = cv2.VideoCapture("assets/videos/sample_1.mp4")
    
    if not videoThread.isOpened():
        print("Error: unable to open source video.")
        return

    width = int(videoThread.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoThread.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = videoThread.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("assets/videos/output_sample.mp4", fourcc, fps, (width, height))

    while True:
        ok, frame = videoThread.read()
        if not ok:
            break

        video_writer.write(frame)
        cv2.imshow('img', frame)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == 27:
            break

    videoThread.release()
    video_writer.release()
    cv2.destroyAllWindows()

readIPWriteTOFile()