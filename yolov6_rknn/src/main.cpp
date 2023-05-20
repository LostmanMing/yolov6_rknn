// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>

#define _BASETSD_H
#include "RgaUtils.h"
#include "im2d.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>




#include "postprocess.h"
#include "rga.h"
#include "rknn_api.h"
#include "spdlog/spdlog.h"

/*-------------------------------------------
                  Functions
-------------------------------------------*/
static void dump_tensor_attr(rknn_tensor_attr *attr) {
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz) {
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp) {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0) {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *) malloc(sz);
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size) {
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main() {

    char *model_name = NULL;
    rknn_context ctx;
    int img_width = 0;
    int img_height = 0;
    int dst_height = 0;
    int dst_weight = 0;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    struct timeval start_time, stop_time;
    int ret;
    // init rga context
    rga_buffer_t src, resize_dst, padding_dst;
    im_rect src_rect, dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    memset(&src, 0, sizeof(src));
    memset(&resize_dst, 0, sizeof(resize_dst));

    spdlog::info("post process config: box_conf_threshold = {}, nms_threshold = {}", box_conf_threshold, nms_threshold);

    model_name = (char*)"../model/RK356X/best_ckpt.rknn";
    char *image_name = (char*)"../model/driver.jpg";
    spdlog::info("Read {} ...", image_name);
    cv::Mat orig_img = cv::imread(image_name);
    if (!orig_img.data) {
        spdlog::error("cv::imread {} fail!",image_name);
        return -1;
    }
    cv::Mat img;
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
    img_width = img.cols;
    img_height = img.rows;
    spdlog::info("img width = {}, img height = {}", img_width, img_height);

    /* Create the neural network */
    spdlog::info("Loading mode...");
    int model_data_size = 0;
    unsigned char *model_data = load_model(model_name, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) {
        spdlog::error("rknn_init error ret={}",ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        spdlog::error("rknn_init error ret={}",ret);
        return -1;
    }
    spdlog::info("sdk version: {} driver version: {}", version.api_version, version.drv_version);
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        spdlog::error("rknn_init error ret={}", ret);
        return -1;
    }
    spdlog::info("model input num: {}, output num: {}\n", io_num.n_input, io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            spdlog::error("rknn_init error ret={}", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }
    //获取输入参数
    int channel = 3;
    int width = 0;
    int height = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        spdlog::info("model is NCHW input fmt");
        channel = input_attrs[0].dims[1];
        height = input_attrs[0].dims[2];
        width = input_attrs[0].dims[3];
    } else {
        spdlog::info("model is NHWC input fmt");
        height = input_attrs[0].dims[1];
        width = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }
    spdlog::info("model input height={}, width={}, channel={}", height, width, channel);

    //设置输入格式
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    // You may not need resize when src resulotion equals to resize_dst resulotion
    void *resize_buf = nullptr;
    void *padding_buf = nullptr;
    if (img_width != width || img_height != height) {
        auto ver = querystring(RGA_VERSION);
        spdlog::info("RGA_VERSION :{} ",ver);
        spdlog::info("resize and padding with RGA!");

        dst_height = img_height > img_width ? height : ((float)width / (float)img_width) * img_height;
        dst_weight = img_width > img_height ? width  : ((float)height / (float)img_height) * img_width;
        resize_buf = malloc(dst_height * dst_weight * channel);
        padding_buf = malloc(height * width * channel);
        memset(resize_buf, 0x00, dst_height * dst_weight * channel);
        memset(padding_buf, 0x00, height * width * channel);

        //resize
        src = wrapbuffer_virtualaddr((void *) img.data, img_width, img_height, _Rga_SURF_FORMAT::RK_FORMAT_RGB_888);
        resize_dst = wrapbuffer_virtualaddr((void *) resize_buf, dst_weight, dst_height, _Rga_SURF_FORMAT::RK_FORMAT_RGB_888);
        ret = imcheck(src, resize_dst, src_rect, dst_rect);
        if (IM_STATUS_NOERROR != ret) {
            spdlog::info("{}, check error! {}", __LINE__, imStrError((IM_STATUS) ret));
            return -1;
        }
        IM_STATUS STATUS = imresize(src, resize_dst);

        //padding
        padding_dst = wrapbuffer_virtualaddr((void *) padding_buf, width, height, _Rga_SURF_FORMAT::RK_FORMAT_RGB_888);
        int top = 0;
        int left = 0;
        int bottom = img_width > img_height ? height - dst_height : 0;
        int right = img_height > img_width ? width - dst_weight : 0;
        ret = imcheck(resize_dst, padding_dst, {}, {});
        if (IM_STATUS_NOERROR != ret) {
            spdlog::error("{}, check error! {}", __LINE__, imStrError((IM_STATUS) ret));
            return -1;
        }
        ret = immakeBorder(resize_dst, padding_dst, top, bottom, left, right, IM_BORDER_REFLECT);
        if (IM_STATUS_SUCCESS != ret) {
            spdlog::error("{}, check error! {}", __LINE__, imStrError((IM_STATUS) ret));
            return -1;
        }
        // for debug
        //cv::Mat resize_img(cv::Size(width, height), CV_8UC3, padding_buf);
        //cv::imwrite("/mnt/mmc/zgm/rknn/rknn_yolov5_demo/model/resize_input.jpg", resize_img);
        inputs[0].buf = padding_buf;
    } else {
        inputs[0].buf = (void *) img.data;
    }

    float size_scale = std::min(width / (img.cols*1.0), height / (img.rows*1.0));
    std::vector<int> fpn_strides{8, 16, 32};  //三种尺度的stride
    std::vector<AnchorPoints> anchor_points;  //手动造anchor
    std::vector<float> stride_tensor;         //stride对齐
    generateAnchors(fpn_strides,anchor_points,stride_tensor);   //generate Anchors

    gettimeofday(&start_time, NULL);
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].want_float = 1;
    }

    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);


    //yolov6-postprocess
    //sacle


    std::vector<Object> objects;  //结果
    decode_outputs((float *)outputs[0].buf, 8400 * 7, objects, size_scale, img_width, img_height,anchor_points,stride_tensor);
    gettimeofday(&stop_time, NULL);
    spdlog::info("once run use {} ms", (__get_us(stop_time) - __get_us(start_time)) / 1000);
    draw_objects(orig_img,objects);


    deinitPostProcess();

    // release
    ret = rknn_destroy(ctx);
    if (model_data) {free(model_data);}
    if (resize_buf) {free(resize_buf);}
    if (padding_buf) {free(padding_buf);}
    return 0;
}
