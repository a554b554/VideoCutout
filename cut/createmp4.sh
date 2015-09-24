#########################################################################
# File Name: createmp4.sh
# Author: DT
# mail: wangliwu@zju.edu.cn
# Created Time: Wed May 20 16:33:20 2015
#########################################################################
./ffmpeg -framerate 24 -i ./TMM/%d.jpg -c:v libx264 -vf "fps=24,format=yuv420p" output.mp4
