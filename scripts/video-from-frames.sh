run_name="GTA5_pixmatch-2021-03-25-12-17-50" # "GTA5_source"
framerate=30

# First sequence
ffmpeg -r $framerate -f image2 -s 1280x640 -i "/home/luke/projects/experiments/pixmatch/tmp/demoVideo_outputs/${run_name}/stuttgart_00_000000_000%03d_leftImg8bit.png" -vcodec libx264 -crf 10 -pix_fmt yuv420p "/home/luke/projects/experiments/pixmatch/tmp/demoVideo_videos/${run_name}-stuttgart_00.mp4"

# Second sequence
ffmpeg -r $framerate -f image2 -s 1280x640 -start_number 3500 -i "/home/luke/projects/experiments/pixmatch/tmp/demoVideo_outputs/${run_name}/stuttgart_01_000000_00%04d_leftImg8bit.png" -vcodec libx264 -crf 10 -pix_fmt yuv420p "/home/luke/projects/experiments/pixmatch/tmp/demoVideo_videos/${run_name}-stuttgart_01.mp4"

# # Third sequence
ffmpeg -r $framerate -f image2 -s 1280x640 -start_number 5100 -i "/home/luke/projects/experiments/pixmatch/tmp/demoVideo_outputs/${run_name}/stuttgart_02_000000_00%04d_leftImg8bit.png" -vcodec libx264 -crf 10 -pix_fmt yuv420p "/home/luke/projects/experiments/pixmatch/tmp/demoVideo_videos/${run_name}-stuttgart_02.mp4"