#include "yolo.h"

namespace algorithms
{
    // 注释部分，接近python版本yolov5,不要删除
    // onnx模型导出为 dyn shape，可以启用。
    /*
    // at::Tensor scale_boxes(cv::Size img1, at::Tensor boxes, cv::Size img0)
    // {
    //     float img1_w = img1.width;
    //     float img1_h = img1.height;

    //     float img0_w = img0.width;
    //     float img0_h = img0.height;

    //     float gain = std::min(img1_h / img0_h, img1_w / img0_w);
    //     float pad_w = (img1_w - img0_w * gain) / 2;
    //     float pad_h = (img1_h - img0_h * gain) / 2;

    //     boxes.select(1, 0) -= pad_w;    // x padding
    //     boxes.select(1, 2) -= pad_w;    // x padding
    //     boxes.select(1, 1) -= pad_h;    // y padding
    //     boxes.select(1, 3) -= pad_h;    // y padding
    //     std::cout << "------------------------------------" << std::endl;
    //     boxes.slice(1,0,4) /= gain;
    //     boxes = clip_boxes(boxes, img0);
    //     return boxes;
    // }


    // std::vector<float> letterbox(const cv::Mat& src, cv::Mat& dst, const cv::Size& new_size=cv::Size(640, 640),
    //  const cv::Scalar color=cv::Scalar(114, 114, 114), bool align=true, bool scaleFill=false, bool scaleup=true, int stride=32)
    //  {
    //     // Resize and pad image while meeting stride-multiple constraints
    //     cv::Size c_size(src.cols, src.rows);  // current shape [width, height]

    //     // Scale ratio (new / old)
    //     float r = std::min(1.0 * new_size.height / c_size.height, 1.0 * new_size.width / c_size.width);
    //     // if(!scaleup)   // only scale down, do not scale up (for better val mAP)
    //     //     r = std::min(r,1.0);

    //     float ratio_w = r;
    //     float ratio_h = r;

    //     int new_unpad_w = std::round(c_size.width * r);
    //     int new_unpad_h = std::round(c_size.height * r);
    //     int dw, dh;  //  wh padding
    //     dw = new_size.width - new_unpad_w;
    //     dh = new_size.height - new_unpad_w;

    //     if (align)   //  minimum rectangle
    //     {
    //         dw = dw % stride;
    //         dh = dh % stride;
    //     }
    //     else if(scaleFill)
    //     {
    //         dw = dh = 0;
    //         new_unpad_w = new_size.width;
    //         new_unpad_h = new_size.height;
    //         ratio_w = new_size.width / c_size.width;
    //         ratio_h = new_size.height / c_size.height;
    //     }

    //     dw /= 2.0;
    //     dh /= 2.0;
    //     cv::Size new_unpad(new_unpad_w, new_unpad_h);
    //     if(c_size != new_unpad)
    //         {
    //             cv::resize(src,dst,new_unpad,0,0,cv::INTER_LINEAR);
    //         }

    //     int top = int(std::round(dh - 0.1));
    //     int bottom = int(std::round(dh + 0.1));
    //     int left = int(std::round(dw - 0.1));
    //     int right = int(std::round(dw + 0.1));
    //     cv::copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    //     std::vector<float> pad_info{static_cast<float>(r), static_cast<float>(left), static_cast<float>(top)};
    //     return pad_info;
    //  }

    // // Clip boxes (xyxy) to image shape (height, width)
    // void clip_boxes(at::Tensor &boxes,const cv::Size &img)
    // {
    //     boxes.slice(1,0).clamp_(0, img.width);
    //     boxes.slice(1,1).clamp_(0, img.height);
    //     boxes.slice(1,2).clamp_(0, img.width);
    //     boxes.slice(1,3).clamp_(0, img.height);
    // }

    */

    void scale_boxes(at::Tensor &boxes, const float &pad_w, const float &pad_h, const float &scale, const cv::Size &img)
    {
        boxes.select(1, 0) -= pad_w;
        boxes.select(1, 2) -= pad_w;
        boxes.select(1, 1) -= pad_h;
        boxes.select(1, 3) -= pad_h;
        boxes.slice(1, 0, 4) /= scale;

        boxes.select(1, 0).clamp_(0, img.width);
        boxes.select(1, 1).clamp_(0, img.height);
        boxes.select(1, 2).clamp_(0, img.width);
        boxes.select(1, 3).clamp_(0, img.height);
    }

    std::vector<float> letterbox(const cv::Mat &src, cv::Mat &dst, const cv::Size &out_size)
    {
        auto in_h = static_cast<float>(src.rows);
        auto in_w = static_cast<float>(src.cols);
        float out_h = out_size.height;
        float out_w = out_size.width;

        float scale = std::min(out_w / in_w, out_h / in_h);

        int mid_h = static_cast<int>(in_h * scale);
        int mid_w = static_cast<int>(in_w * scale);

        cv::resize(src, dst, cv::Size(mid_w, mid_h));

        int top = (static_cast<int>(out_h) - mid_h) / 2;
        int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
        int left = (static_cast<int>(out_w) - mid_w) / 2;
        int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

        cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
        return pad_info;
    }

    /*
        Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    */
    at::Tensor xywh2xyxy(at::Tensor x)
    {
        at::Tensor y =
            x.new_empty(x.sizes(), x.options());

        y.select(1, 0) =
            (x.select(1, 0) - x.select(1, 2).div(2));
        y.select(1, 1) =
            (x.select(1, 1) - x.select(1, 3).div(2));
        y.select(1, 2) =
            (x.select(1, 0) + x.select(1, 2).div(2));
        y.select(1, 3) =
            (x.select(1, 1) + x.select(1, 3).div(2));

        return y;
    }

    /*
        Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
        Returns:
             list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    */
    at::Tensor non_max_suppression(at::Tensor prediction, float conf_thres, float iou_thres, int nm)
    {
        auto device = prediction.device();

        auto bs = prediction.size(0);          // batch size
        auto nc = prediction.size(2) - nm - 5; // number of classes
         //std::cout << "----------------- start1" << std::endl;
        auto xc = prediction.select(2, 4).gt(conf_thres).unsqueeze(2); // candidates
         //std::cout << "----------------- end" << std::endl;
         //std::cout << xc.sizes() << std::endl;

        auto mi = 5 + nc;
        std::vector<at::Tensor> outputs;
        for (int xi = 0; xi < bs; ++xi)
        {
             //std::cout << "----------------- start2" << std::endl;
            auto x = prediction[xi];
            auto det = x.masked_select(xc[xi]).reshape({-1, 5 + nc + nm}); // # confidence
             //std::cout << "----------------- start3" << std::endl;
            if (!det.size(0))
                continue;

             //std::cout << det.sizes() << std::endl;

             //std::cout << det.slice(1,5).sizes() << std::endl;
             //std::cout << det.select(1,4).sizes() << std::endl;
            // Compute conf
            det.slice(1, 5) *= det.select(1, 4).unsqueeze(1); // conf = obj_conf * cls_conf
             //std::cout << "----------------- start4" << std::endl;

            // Box/Mask
            auto box = xywh2xyxy(det.slice(1, 0, 4)); // center_x, center_y, width, height) to (x1, y1, x2, y2)
            auto mask = det.slice(1, mi);
             //std::cout << "----------------- start" << std::endl;
             //std::cout << mask.sizes() << std::endl;
             //std::cout << "----------------- end" << std::endl;
            // Detections matrix nx6 (xyxy, conf, cls)
            // best class only
            auto [conf, j] = det.slice(1, 5, mi).max(1, true);
            det = at::cat({box, conf, j.to(at::kFloat), mask}, 1);

            // Batched NMS
            auto max_wh = 7680; // # (pixels) maximum box width and height
            // auto max_nms = 30000; // # maximum number of boxes into torchvision.ops.nms()

             //std::cout << "--------------------------" << std::endl;
            auto c = det.slice(1, 5, 6) * max_wh; //  classes
             //std::cout << c.sizes() << std::endl;
            auto boxes = det.slice(1, 0, 4) + c; //  boxes (offset by class)
             //std::cout << boxes.sizes() << std::endl;
            auto scores = det.select(1, 4); //  scores
             //std::cout << scores.sizes() << std::endl;
            auto i = vision::ops::nms(boxes, scores, iou_thres); // NMS
            auto a = det.index_select(0, i);
            outputs.emplace_back(a.unsqueeze(0));
        }

        auto nOut = outputs.size();
        if (nOut > 0)
            return at::cat(outputs, 0);
        else
            return at::empty({0,1,6});
    }

    template <class Type>
    Type string2Num(const std::string &str)
    {
        std::istringstream iss(str);
        Type num;
        iss >> std::hex >> num;
        return num;
    }

    /*
        return [r g b] * n
    */
    at::Tensor generator_colors(int num)
    {
        std::vector<std::string> hexs = {"FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
                                         "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7"};

        std::vector<int> tmp;
        for (int i = 0; i < num; ++i)
        {
            int r = string2Num<int>(hexs[i].substr(0, 2));
            // std::cout << r << std::endl;
            int g = string2Num<int>(hexs[i].substr(2, 2));
            // std::cout << g << std::endl;
            int b = string2Num<int>(hexs[i].substr(4, 2));
            // std::cout << b << std::endl;
            tmp.emplace_back(r);
            tmp.emplace_back(g);
            tmp.emplace_back(b);
        }
        return at::from_blob(tmp.data(), {(int)tmp.size()}, at::TensorOptions(at::kInt));
    }

    /*
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Vectorized by Chong (thanks Chong).

        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    */

    at::Tensor crop_mask(at::Tensor masks, at::Tensor boxes)
    {
        // std::cout << "masks:" << masks.sizes() << std::endl;
        // std::cout << "boxes:" << boxes.sizes() << std::endl;
        auto n = masks.size(0), h = masks.size(1), w = masks.size(2);
        auto v_list = at::chunk(boxes.unsqueeze(2), 4, 1);
        auto x1 = v_list[0], y1 = v_list[1], x2 = v_list[2], y2 = v_list[3];
        // std::cout << "------------------------------------------" << std::endl;
        // std::cout << "x1:" << x1.sizes() << std::endl;
        // std::cout << "y1:" << y1.sizes() << std::endl;
        // std::cout << "x2:" << x2.sizes() << std::endl;
        // std::cout << "y2:" << y2.sizes() << std::endl;
        // std::cout << "------------------------------------------" << std::endl;
        auto r = at::arange(w, boxes.options()).unsqueeze(0).unsqueeze(0);
        auto c = at::arange(w, boxes.options()).unsqueeze(0).unsqueeze(2);
        // std::cout << "------------------------------------------" << std::endl;
        // std::cout << "r:" << r.sizes() << std::endl;
        // std::cout << "c:" << c.sizes() << std::endl;
        // std::cout << "------------------------------------------" << std::endl;
        // std::cout << "A:" << r.ge(x1).sizes() << std::endl;
        // std::cout << "B:" << r.lt(x2).sizes() << std::endl;
        // std::cout << "C:" << c.ge(y1).sizes() << std::endl;
        // std::cout << "D:" << c.lt(y2).sizes() << std::endl;
        // std::cout << "------------------------------------------" << std::endl;
        // >= * < * >= * <
        return masks * (r.ge(x1) * r.lt(x2) * c.ge(y1) * c.lt(y2));
    }

    /*
        Crop before upsample.
        proto_out: [mask_dim, mask_h, mask_w]
        out_masks: [n, mask_dim], n is number of masks after nms
        bboxes: [n, 4], n is number of masks after nms
        shape:input_image_size, (h, w)

        return: h, w, n
    */
    at::Tensor process_mask(at::Tensor protos, at::Tensor masks_in, at::Tensor bboxes, at::IntArrayRef shape, bool upsample)
    {
        // std::cout << "--------------------------------1" << std::endl;
        // std::cout << "masks_in:" << masks_in.sizes() << std::endl;
        // std::cout << "bboxes:" << bboxes.sizes() << std::endl;
        auto c = protos.size(0);
        auto mh = protos.size(1);
        auto mw = protos.size(2);
        auto ih = shape[0];
        auto iw = shape[1];
        auto p = protos.to(at::kFloat).view({c, -1}); // CHW
        // std::cout << "p:" << p.sizes() << std::endl;
        auto masks = masks_in.matmul(p).sigmoid().view({-1, mh, mw});
        // std::cout << "--------------------------------2" << std::endl;

        auto downsampled_bboxes = bboxes.clone();
        downsampled_bboxes.select(1, 0) *= 1.0 * mw / iw;
        downsampled_bboxes.select(1, 2) *= 1.0 * mw / iw;
        downsampled_bboxes.select(1, 3) *= 1.0 * mh / ih;
        downsampled_bboxes.select(1, 1) *= 1.0 * mh / ih;
        // std::cout << "--------------------------------3" << std::endl;

        masks = crop_mask(masks, downsampled_bboxes);
        // std::cout << "--------------------------------4" << std::endl;
        if (upsample)
        {
            namespace F = torch::nn::functional;
            masks = F::interpolate(masks.unsqueeze(0), F::InterpolateFuncOptions().size(std::vector<int64_t>({shape[0], shape[1]})).mode(torch::kBilinear).align_corners(false));
            // std::cout << "--------------------------------5" << std::endl;
        }
        return masks.gt(0.5).squeeze(0);
    }

    /*
        img1_shape: model input shape, [h, w]
        img0_shape: origin pic shape, [h, w, 3]
        masks: [h, w, num]
    */
    at::Tensor scale_image(at::IntArrayRef im1_shape, at::Tensor masks, at::IntArrayRef im0_shape, const float &pad_w, const float &pad_h, const float &scale)
    {
        // std::cout << "scale_image:" << std::endl;
        int top = static_cast<int>(pad_h), left = static_cast<int>(pad_w);
        int bottom = im1_shape[0] - top, right = im1_shape[1] - left;
        masks = masks.slice(0, top, bottom).slice(1, left, right);
        // std::cout << "masks: " << masks.sizes() << std::endl;
        namespace F = torch::nn::functional;
        masks = F::interpolate(masks.permute({2, 0, 1}).unsqueeze(0), F::InterpolateFuncOptions().size(std::vector<int64_t>({im0_shape[0], im0_shape[1]})).mode(torch::kBilinear).align_corners(false));
        // std::cout << "masks: " << masks.sizes() << std::endl;
        return masks.squeeze(0).permute({1, 2, 0}).contiguous();
        // return masks.squeeze(0);
    }

    /*
        Plot masks at once.
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
    */
    at::Tensor plot_masks(at::Tensor masks, at::IntArrayRef im0_shape, const float &pad_w, const float &pad_h, const float &scale, at::Tensor im_gpu, float alpha)
    {
        // std::cout << "Plotting masks: " << std::endl;
        int n = masks.size(0);
        auto colors = generator_colors(n);
        colors = colors.to(masks.device()).to(at::kFloat).div(255).reshape({-1, 3}).unsqueeze(1).unsqueeze(2);
        // std::cout << "colors: " << colors.sizes() << std::endl;
        masks = masks.unsqueeze(3);
        // std::cout << "masks: " << masks.sizes() << std::endl;
        auto masks_color = masks * (colors * alpha);
        // std::cout << "masks_color: " << masks_color.sizes() << std::endl;
        auto inv_alph_masks = (1 - masks * alpha);
        inv_alph_masks = inv_alph_masks.cumprod(0);
        // std::cout << "inv_alph_masks: " << inv_alph_masks.sizes() << std::endl;

        auto mcs = masks_color * inv_alph_masks;
        mcs = mcs.sum(0) * 2;
        // std::cout << "mcs: " << mcs.sizes() << std::endl;
        im_gpu = im_gpu.flip({0});
        // std::cout << "im_gpu: " << im_gpu.sizes() << std::endl;
        im_gpu = im_gpu.permute({1, 2, 0}).contiguous();
        // std::cout << "im_gpu: " << im_gpu.sizes() << std::endl;
        im_gpu = im_gpu * inv_alph_masks[-1] + mcs;
        // std::cout << "im_gpu: " << im_gpu.sizes() << std::endl;
        auto im_mask = (im_gpu * 255);
        auto result = scale_image(im_gpu.sizes(), im_mask, im0_shape, pad_w, pad_h, scale);
        // std::cout << "Plotting masks: " << std::endl;
        return result;
    }

    std::vector<std::string> read_names(const std::string filename)
    {
        std::vector<std::string> names;
        std::ifstream infile(filename);
        // assert(stream.is_open());

        std::string line;
        while (std::getline(infile, line))
        {
            names.emplace_back(line);
        }
        return names;
    }

}