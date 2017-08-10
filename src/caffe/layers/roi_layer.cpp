// ------------------------------------------------------------------
// patch sample learning
// for SPN
// Sifei Liu
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/roi_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROILayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIParameter roi_param = this->layer_param_.roi_param();
  CHECK_GT(roi_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_param.pooled_h();
  pooled_width_ = roi_param.pooled_w();
  //spatial_scale_ = roi_pool_param.spatial_scale();
  //LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROILayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  mask_.Reshape(bottom[0]->num(), channels_, height_, width_);
}

template <typename Dtype>
void ROILayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data(); 
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int bottom_count = bottom[0]->count();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  // mask to record sampled patches
  int* argmax_data = mask_.mutable_cpu_data();
  caffe_set(bottom_count, 0, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = bottom_rois[1];
    int roi_start_h = bottom_rois[2];
    int roi_end_w = bottom_rois[3];
    int roi_end_h = bottom_rois[4];
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);

    CHECK_EQ(roi_width,pooled_width_) << "roi width should be the same with the pre defined width.";
    CHECK_EQ(roi_height,pooled_height_) << "roi height should be the same with the pre defined height.";


    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);
    int* arg_batch_data = argmax_data + mask_.offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c)
    {
      for (int h = roi_start_h; h < roi_end_h; ++h)
        for (int w = roi_start_w; w < roi_end_w; ++w)
        {
          const int bottom_index = h * width_ + w;
          const int top_index = (h - roi_start_h) * roi_width + (w - roi_start_w);
          top_data[top_index] = batch_data[bottom_index];
          arg_batch_data[bottom_index] = 1;
        }
        // Increment all data pointers by one channel
        batch_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        arg_batch_data += mask_.offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void ROILayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ROILayer);
#endif

INSTANTIATE_CLASS(ROILayer);
REGISTER_LAYER_CLASS(ROI);

}  // namespace caffe