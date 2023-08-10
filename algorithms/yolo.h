#ifndef YOLO_H
#define YOLO_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <torchvision/ops/ops.h>

namespace algorithms
{

     void scale_boxes(at::Tensor &boxes, const float &pad_w, const float &pad_h, const float &scale, const cv::Size &img);

     std::vector<float> letterbox(const cv::Mat &src, cv::Mat &dst, const cv::Size &out_size);

    /*
        Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    */
     at::Tensor xywh2xyxy(at::Tensor x);

    /*
        Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    */
     at::Tensor non_max_suppression(at::Tensor prediction, float conf_thres = 0.45, float iou_thres = 0.25, int nm = 0);

    /*
        return [r g b] * n
    */
     at::Tensor generator_colors(int num);

    /*
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Vectorized by Chong (thanks Chong).

        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    */

     at::Tensor crop_mask(at::Tensor masks, at::Tensor boxes);

    /*
        Crop before upsample.
        proto_out: [mask_dim, mask_h, mask_w]
        out_masks: [n, mask_dim], n is number of masks after nms
        bboxes: [n, 4], n is number of masks after nms
        shape:input_image_size, (h, w)

        return: h, w, n
    */
     at::Tensor process_mask(at::Tensor protos, at::Tensor masks_in, at::Tensor bboxes, at::IntArrayRef shape, bool upsample = false);

    /*
        img1_shape: model input shape, [h, w]
        img0_shape: origin pic shape, [h, w, 3]
        masks: [h, w, num]
    */
     at::Tensor scale_image(at::IntArrayRef im1_shape, at::Tensor masks, at::IntArrayRef im0_shape, const float &pad_w, const float &pad_h, const float &scale);

    /*
        Plot masks at once.
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
    */
     at::Tensor plot_masks(at::Tensor masks, at::IntArrayRef im0_shape, const float &pad_w, const float &pad_h, const float &scale, at::Tensor im_gpu, float alpha = 0.5);
}

#endif