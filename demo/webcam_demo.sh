#!/usr/bin/env bash
base_path=$1
video_path=$2
echo ${base_path}
echo ${video_path}

# set variable
config_file=configs/${base_path}.py
work_path=work_dirs/${base_path}
ckpt_file=${work_path}/latest.pth
video_file=${work_path}/test_video.mp4
video_slim_file=${work_path}/test_video_slim.mp4

python demo/webcam_demo.py ${config_file} \
    ${ckpt_file} \
    --camera-id ${video_path} --out-video ${video_file} --rot 1

# slim output
ffmpeg -i ${video_file} -codec:audio aac -b:audio 128k -codec:video libx264 -crf 23 ${video_slim_file}
