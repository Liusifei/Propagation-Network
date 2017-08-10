// ------------------------------------------------------------------
// patch sample learning
// for SPN
// Sifei Liu
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/roi_layer.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void ROIForward(const int nthreads, const Dtype* bottom_data,
    const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the top
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = bottom_rois[1];
    int roi_start_h = bottom_rois[2];
    // int roi_end_w = bottom_rois[3];
    // int roi_end_h = bottom_rois[4];
    // Force malformed ROIs to be 1x1
    // int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    // int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    // CHECK(roi_width == pooled_width) << "roi width should be the same with the pre defined width.";
    // CHECK(roi_height == pooled_height) << "roi height should be the same with the pre defined height.";

    int bottom_index = roi_batch_ind * channels * height * width + c * height * width + (roi_start_h + ph) * width + (roi_start_w + pw);

    top_data[index] = bottom_data[bottom_index];
    argmax_data[bottom_index] = 1;

  }
}

template <typename Dtype>
void ROILayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = mask_.mutable_gpu_data();
  int count = top[0]->count();
  int bottom_count = bottom[0]->count();
  caffe_gpu_set(bottom_count, 0, argmax_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void ROIBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the top
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;


    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = bottom_rois[1];
    int roi_start_h = bottom_rois[2];

    int bottom_index = roi_batch_ind * channels * height * width + c * height * width + (roi_start_h + ph) * width + (roi_start_w + pw);
    if (argmax_data[bottom_index] != 0) {
      bottom_diff[bottom_index] = top_diff[index];
    }
  }
}

template <typename Dtype>
void ROILayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int bottom_count = bottom[0]->count();
  caffe_gpu_set(bottom_count, Dtype(0.), bottom_diff);
  const int count = top[0]->count();
  const int* argmax_data = mask_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, argmax_data, channels_, height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROILayer);

}  // namespace caffe