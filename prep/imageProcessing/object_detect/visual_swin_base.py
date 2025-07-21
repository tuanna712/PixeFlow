from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.core import DatasetEnum

if __name__ == '__main__':
    config_file = '../../../model/Co-DETR/checkpoint/co_deformable_detr_swin_base_3x_coco.py'
    checkpoint_file = '../../../model/Co-DETR/checkpoint/co_deformable_detr_swin_base_3x_coco.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, DatasetEnum.COCO, device='cuda:0')

    # test a single image
    img = 'demo/demo.jpg'
    result = inference_detector(model, img)
    print(result)
    # show the results
    show_result_pyplot(model, img, result)