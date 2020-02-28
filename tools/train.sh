# old

#base_paths=(helmet/concat/faster_rcnn_r50_fpn_2x helmet/concat/ga_retinanet_r50_caffe_fpn_2x helmet/concat/retinanet_r50_fpn_2x)

#for base_path in helmet/concat/faster_rcnn_r50_fpn_2x helmet/concat/ga_retinanet_r50_caffe_fpn_2x helmet/concat/retinanet_r50_fpn_2x

#for base_path in helmet/merge/faster_rcnn_r50_fpn_2x helmet/merge/ga_retinanet_r50_caffe_fpn_2x helmet/merge/retinanet_r50_fpn_2x
#for base_path in helmet/merge/retinanet_r50_fpn_2x helmet/merge/ga_retinanet_r50_caffe_fpn_2x
#for base_path in helmet/merge/faster_rcnn_mobilenetv2_32_fpn_2x helmet/merge/faster_rcnn_mobilenetv2_64_fpn_2x helmet/merge/faster_rcnn_mobilenetv2_128_fpn_2x
# for base_path in helmet/merge/retinanet_mobilenetv2_128_fpn_1x helmet/merge/retinanet_mobilenetv2_128_fpn_1x
# do
#base_path=old_helmet/concat/retinanet_r50_fpn_2x
#base_path=old_helmet/concat/faster_rcnn_r50_fpn_2x
#old_helmet/concat/faster_rcnn_r50_fpn_2x

for config_file in "$@"
do
echo ${config_file}
base_path=`echo ${config_file%.*}`
base_path=`echo ${base_path#*/}`
echo ${base_path}
#work_path=`echo ${config_file/configs/work_dirs}`
#work_path=`echo ${work_path%.*}`
#echo ${work_path}
config_file=configs/${base_path}.py
work_path=work_dirs/${base_path}
ckpt_file=${work_path}/latest.pth
test_output_file=${work_path}/test.pkl
map_file=${work_path}/map.txt
flops_file=${work_path}/model_flops.txt

mkdir -p ${work_path}
## get_flops
python tools/get_flops.py ${config_file} | tee ${flops_file}
## train
python tools/train.py ${config_file}
## publish
python tools/publish_model.py ${ckpt_file} ${ckpt_file}
#
## plot
python tools/analyze_logs.py plot_curve \
${work_path}/*.log.json \
--keys loss_cls loss_bbox loss lr \
--legend loss_cls loss_bbox loss lr \
--out ${work_path}/losses.pdf

## test
python tools/test.py ${config_file} \
${ckpt_file} \
--out ${test_output_file}

# eval
for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
  python tools/voc_eval.py ${test_output_file} \
  ${config_file} --iou-thr ${i} | tee -a ${map_file}
done

done