#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/isomap_manifold_learning_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void IsomapManifoldLearningLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  //////////////////////////////////////////////////////////////////////////
  inter_weight_ = this->layer_param_.isomap_manifold_learning_loss_param().inter_weight();
  intra_weight_ = this->layer_param_.isomap_manifold_learning_loss_param().intra_weight();
  num_output_ = this->layer_param_.isomap_manifold_learning_loss_param().num_output();
  class_k_ = this->layer_param_.isomap_manifold_learning_loss_param().class_k();
  //////////////////////////////////////////////////////////////////////////
  //初始化各个不同类中心点
  const int axis = bottom[0]->CanonicalAxisIndex(
	  this->layer_param_.isomap_manifold_learning_loss_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis); 
  /*Data_K_ = bottom[2]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
	  LOG(INFO) << "Skipping parameter initialization";
  }
  else {
	  this->blobs_.resize(1);
	  // Intialize the weight
	  vector<int> center_shape(3);
	  center_shape[0] = num_output_;
	  center_shape[1] = class_k_;
	  center_shape[2] = Data_K_;
	  this->blobs_[0].reset(new Blob<Dtype>(center_shape));
	  // fill the weights
	  shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(
		  this->layer_param_.isomap_manifold_learning_loss_param().center_filler()));
	  center_filler->Fill(this->blobs_[0].get());

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  //////////////////////////////////////////////////////////////////////////*/
  M_ = num_output_;
  N_ = bottom[0]->num();  
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  class_num_.Reshape(N_, 1, 1, 1);
  distance_sample_k_.Reshape(N_, class_k_, 1, 1);
  //dist_class_.Reshape(Data_K_, 1, 1, 1);
  mend_.Reshape(1, 1, 1, 1);
  //dist_center_.Reshape(Data_K_, 1, 1, 1);
  dist_intra_.Reshape(K_, 1, 1, 1);
  dist_inter_.Reshape(K_, 1, 1, 1);
  dot_.Reshape(N_, N_, 1, 1);
  distance_inter_.Reshape(1, 1, 1, 1);
  distance_inter_1_.Reshape(1, 1, 1, 1);
  distance_max_min_.Reshape(1, 1, 1, 1);
  distance_max_min_1_.Reshape(1, 1, 1, 1);
  ones_.Reshape(bottom[0]->num(), 1, 1, 1);  // n by 1 vector of ones.
  sample_max_min_1_.Reshape(1, 1, 1, 1);
  sample_max_min_2_.Reshape(1, 1, 1, 1);
  sequence_sample_.Reshape(N_, 1, 1, 1);  
  //variation_sum_data_.Reshape(Data_K_, 1, 1, 1);
  //one_.Reshape(1, 1, 1, 1);
  //one_.mutable_cpu_data()[0] = 1;
  sub_class_num_.Reshape(num_output_, class_k_, 1, 1);  
  sum_sub_class_num_.Reshape(num_output_, class_k_, 1, 1);  
  intra_loss_.Reshape(1, 1, 1, 1);
  for (int i = 0; i < bottom[0]->num(); ++i){
	  ones_.mutable_cpu_data()[i] = Dtype(1);
  }
} 

template <typename Dtype>
void IsomapManifoldLearningLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  /*Dtype* label_1 = bottom[1]->mutable_cpu_data();
  for (int i = 0; i < bottom[1]->num(); i++){
	  int mend = (int) label_1[i] / class_k_;
	  label_1[i] = (float) mend;
  }*/
  //////////////////////////////////////////////////////////////////////////
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  //const Dtype* center = this->blobs_[0]->cpu_data();
  /*//////////////////////////////////////////////////////////////////////////

  //compute the distance of samples to the different centers in the training batch计算所有batch样本距不同中心点的距离
  for (int j = 0; j < N_; j++){
      const int label_value = static_cast<int>(label[j]);
	  for (int k = 0; k < class_k_; k++){
		  caffe_sub(Data_K_, center + label_value*class_k_*Data_K_ + k*Data_K_, bottom[2]->cpu_data() + j*Data_K_, dist_class_.mutable_cpu_data());
		  mend_.mutable_cpu_data()[0]=caffe_cpu_dot(Data_K_, dist_class_.cpu_data(), dist_class_.cpu_data());
		  caffe_copy(1, mend_.cpu_data(), distance_sample_k_.mutable_cpu_data() + j*class_k_ + k);
	  }
  }
  //////////////////////////////////////////////////////////////////////////
  // compute the sub-class of different samples
  Dtype* class_num = class_num_.mutable_cpu_data();
  caffe_set(N_, (Dtype)0., class_num_.mutable_cpu_data());
  for (int i = 0; i < N_; i++){
	  caffe_copy(1, distance_sample_k_.cpu_data() + i*class_k_, mend_.mutable_cpu_data());
	  Dtype tend = mend_.cpu_data()[0];
	  for (int j = 1; j < class_k_; j++){
		  caffe_copy(1, distance_sample_k_.cpu_data() + i*class_k_ + j, mend_.mutable_cpu_data());
		  if (tend>mend_.cpu_data()[0]){
			  tend = mend_.cpu_data()[0];
			  class_num[i] = j;
		  }
	  }
  }*/
  //////////////////////////////////////////////////////////////////////////
  //compute inter-sample distance
  const int channels = bottom[0]->channels();
  for (int i = 0; i < bottom[0]->num(); i++){
	  dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels, bottom[0]->cpu_data() + (i*channels), bottom[0]->cpu_data() + (i*channels));
  }
  const Dtype* bottom_data1 = bottom[0]->cpu_data();
  const Dtype* bottom_data2 = bottom[0]->cpu_data();

  Dtype dot_scaler(-2.0);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, N_, K_, dot_scaler, bottom_data1, bottom_data2, (Dtype)0., dot_.mutable_cpu_data());

  // add ||x_i||^2 to all elements in row i
  for (int i = 0; i<N_; i++){
	  caffe_axpy(N_, dist_sq_.cpu_data()[i], ones_.cpu_data(), dot_.mutable_cpu_data() + i*N_);
  }

  // add the norm vector to row i
  for (int i = 0; i<N_; i++){
	  caffe_axpy(N_, Dtype(1.0), dist_sq_.cpu_data(), dot_.mutable_cpu_data() + i*N_);
  }

  //compute gradient
  Dtype* bout = bottom[0]->mutable_cpu_diff();
  // zero initialize bottom[0]->mutable_cpu_diff();
  for (int i = 0; i<N_; i++){
	  caffe_set(K_, Dtype(0.0), bout + i*K_);
  }
  //////////////////////////////////////////////////////////////////////////
  // number of samples of different sub-classes in each class
  caffe_set(num_output_*class_k_, (Dtype)0., sub_class_num_.mutable_cpu_data());
  for (int i = 0; i < N_; i++){
	  const int class_sam = static_cast<int>(label[i]);
	  sub_class_num_.mutable_cpu_data()[class_sam] += 1;
	  //const int sub_class_sam = static_cast<int>(class_num_.cpu_data()[i]);
	  //caffe_axpy(1, (Dtype)1, one_.cpu_data(), sub_class_num_.mutable_cpu_data() + class_sam*class_k_ + sub_class_sam);
  }
  //
  int mend = 0;
  Dtype* sequence_sample = sequence_sample_.mutable_cpu_data();
  for (int i = 0; i < num_output_; i++){
	  for (int j = 0; j < class_k_; j++){
		  for (int n = 0; n < N_; n++){
			  const int class_sam = static_cast<int>(label[n]);
			  //const int sub_class_sam = static_cast<int>(class_num_.cpu_data()[n]);
			  if (class_sam == (i *class_k_ + j)){
				  sequence_sample[mend] = n;
				  mend = mend + 1;
			  }
		  }
	  }
  }
  for (int i = 0; i < num_output_*class_k_; i++){
	  if (i == 0){
		  sum_sub_class_num_.mutable_cpu_data()[i] = 0;
	  }
	  else{
		  sum_sub_class_num_.mutable_cpu_data()[i] = sub_class_num_.cpu_data()[i-1] + sum_sub_class_num_.mutable_cpu_data()[i-1];
	  } 
  }
  //////////////////////////////////////////////////////////////////////////
  //intra-class distance
  Dtype intra_loss(0.0);
  for (int i = 0; i < N_; i++){
	  for (int j = i + 1; j < N_; j++){
		  const int class_sam = static_cast<int>(label[i]);
		  //const int sub_class_sam = static_cast<int>(class_num_.cpu_data()[i]);
		  const int class_sam_1 = static_cast<int>(label[j]);
		  //const int sub_class_sam_1 = static_cast<int>(class_num_.cpu_data()[j]);
		  if (class_sam == class_sam_1){
			  caffe_sub(K_, bottom_data + i*K_, bottom_data + j*K_, dist_intra_.mutable_cpu_data());
			  intra_loss_.mutable_cpu_data()[0] = caffe_cpu_dot(K_, dist_intra_.cpu_data(), dist_intra_.cpu_data());
			  caffe_scal(1, 1/(sub_class_num_.cpu_data() + class_sam)[0], intra_loss_.mutable_cpu_data());
			  intra_loss += intra_loss_.cpu_data()[0];
			  //compute gradient
			  caffe_axpy(K_, (Dtype)2.0 * intra_weight_ / (sub_class_num_.cpu_data() + class_sam)[0], dist_intra_.cpu_data(), bout + i*K_);
			  caffe_axpy(K_, (Dtype)-2.0 * intra_weight_ / (sub_class_num_.cpu_data() + class_sam)[0], dist_intra_.cpu_data(), bout + j*K_);
		  }
		  
	  }  
  }
  //////////////////////////////////////////////////////////////////////////
  /*//inter-class distance: Hausdorff distance \max\min d(x,y)
  Dtype margin = this->layer_param_.isomap_manifold_learning_loss_param().margin();
  Dtype inter_loss(0.0);
  int sample_num_1(0);
  int sample_num_2(0);
  for (int i = 0; i < num_output_; i++){
	  for (int j = 0; j < num_output_; j++){
		  if (j != i){
			  for (int i_k = 0; i_k < class_k_; i_k++){
				  for (int j_k = 0; j_k < class_k_; j_k++){
					  caffe_set(1, (Dtype)0., distance_max_min_.mutable_cpu_data());
					  caffe_set(1, (Dtype)0., distance_max_min_1_.mutable_cpu_data());
					  for (int n = static_cast<int>(sum_sub_class_num_.cpu_data()[i*class_k_ + i_k]); n < static_cast<int>(sum_sub_class_num_.cpu_data()[i*class_k_ + i_k] + sub_class_num_.cpu_data()[i*class_k_ + i_k]); n++){
						  caffe_set(1, (Dtype)0., distance_inter_.mutable_cpu_data());
						  caffe_set(1, (Dtype)0., distance_inter_1_.mutable_cpu_data());
						  for (int nn = static_cast<int>(sum_sub_class_num_.cpu_data()[j*class_k_ + j_k]); nn < static_cast<int>(sum_sub_class_num_.cpu_data()[j*class_k_ + j_k] + sub_class_num_.cpu_data()[j*class_k_ + j_k]); nn++){
								  if (distance_inter_.cpu_data()[0] < 0.0001) {
									  caffe_copy(1, dot_.cpu_data() + static_cast<int>(sequence_sample_.cpu_data()[n] * N_ + sequence_sample_.cpu_data()[nn]), distance_inter_.mutable_cpu_data());
									  sample_num_1 = static_cast<int>(sequence_sample_.cpu_data()[n]);
									  sample_num_2 = static_cast<int>(sequence_sample_.cpu_data()[nn]);
								  }
								  else{
									  caffe_copy(1, dot_.cpu_data() + static_cast<int>(sequence_sample_.cpu_data()[n] * N_ + sequence_sample_.cpu_data()[nn]), distance_inter_1_.mutable_cpu_data());
									  if (distance_inter_1_.cpu_data()[0] < distance_inter_.cpu_data()[0]){
										  caffe_copy(1, distance_inter_1_.cpu_data(), distance_inter_.mutable_cpu_data());
										  sample_num_1 = static_cast<int>(sequence_sample_.cpu_data()[n]);
										  sample_num_2 = static_cast<int>(sequence_sample_.cpu_data()[nn]);
									  }
								  }
						  }
						  if ((distance_max_min_.cpu_data()[0] < 0.0001) || (distance_inter_.cpu_data()[0] > distance_max_min_.cpu_data()[0])){
							  caffe_copy(1, distance_inter_.cpu_data(), distance_max_min_.mutable_cpu_data());
							  sample_max_min_1_.mutable_cpu_data()[0] = sample_num_1;
							  sample_max_min_2_.mutable_cpu_data()[0] = sample_num_2;
						  }
					  }
					  if (distance_max_min_.cpu_data()[0] > 0.0001){
						  inter_loss += margin - distance_max_min_.cpu_data()[0];
						  //compute gradient
						  const int sample_1 = static_cast<int>(sample_max_min_1_.cpu_data()[0]);
						  const int sample_2 = static_cast<int>(sample_max_min_2_.cpu_data()[0]);
						  caffe_sub(K_, bottom_data + sample_1*K_, bottom_data + sample_2*K_, dist_inter_.mutable_cpu_data());
						  caffe_axpy(K_, (Dtype)(-2.0) * inter_weight_, dist_inter_.cpu_data(), bout + sample_1*K_);
						  caffe_axpy(K_, (Dtype)(2.0) * inter_weight_, dist_inter_.cpu_data(), bout + sample_2*K_);
					  }
					  
				  }
			  }
		  }
	  }
  }*/
  //inter-class distance: \min\min d(x,y)
  Dtype margin = this->layer_param_.isomap_manifold_learning_loss_param().margin();
  Dtype inter_loss(0.0);
  int sample_num_1(0);
  int sample_num_2(0);
  for (int i = 0; i < num_output_; i++){
	  for (int j = i+1; j < num_output_; j++){
		  //if (j != i){
		  for (int i_k = 0; i_k < class_k_; i_k++){
			  for (int j_k = 0; j_k < class_k_; j_k++){
				  caffe_set(1, (Dtype)0., distance_max_min_.mutable_cpu_data());
				  caffe_set(1, (Dtype)0., distance_max_min_1_.mutable_cpu_data());
				  for (int n = static_cast<int>(sum_sub_class_num_.cpu_data()[i*class_k_ + i_k]); n < static_cast<int>(sum_sub_class_num_.cpu_data()[i*class_k_ + i_k] + sub_class_num_.cpu_data()[i*class_k_ + i_k]); n++){
					  caffe_set(1, (Dtype)0., distance_inter_.mutable_cpu_data());
					  caffe_set(1, (Dtype)0., distance_inter_1_.mutable_cpu_data());
					  for (int nn = static_cast<int>(sum_sub_class_num_.cpu_data()[j*class_k_ + j_k]); nn < static_cast<int>(sum_sub_class_num_.cpu_data()[j*class_k_ + j_k] + sub_class_num_.cpu_data()[j*class_k_ + j_k]); nn++){
						  if (distance_inter_.cpu_data()[0] < 0.0001) {
							  caffe_copy(1, dot_.cpu_data() + static_cast<int>(sequence_sample_.cpu_data()[n] * N_ + sequence_sample_.cpu_data()[nn]), distance_inter_.mutable_cpu_data());
							  sample_num_1 = static_cast<int>(sequence_sample_.cpu_data()[n]);
							  sample_num_2 = static_cast<int>(sequence_sample_.cpu_data()[nn]);
						  }
						  else{
							  caffe_copy(1, dot_.cpu_data() + static_cast<int>(sequence_sample_.cpu_data()[n] * N_ + sequence_sample_.cpu_data()[nn]), distance_inter_1_.mutable_cpu_data());
							  if (distance_inter_1_.cpu_data()[0] < distance_inter_.cpu_data()[0]){
								  caffe_copy(1, distance_inter_1_.cpu_data(), distance_inter_.mutable_cpu_data());
								  sample_num_1 = static_cast<int>(sequence_sample_.cpu_data()[n]);
								  sample_num_2 = static_cast<int>(sequence_sample_.cpu_data()[nn]);
							  }
						  }
					  }
					  if ((distance_max_min_.cpu_data()[0] < 0.0001) || (distance_inter_.cpu_data()[0] < distance_max_min_.cpu_data()[0])){
						  caffe_copy(1, distance_inter_.cpu_data(), distance_max_min_.mutable_cpu_data());
						  sample_max_min_1_.mutable_cpu_data()[0] = sample_num_1;
						  sample_max_min_2_.mutable_cpu_data()[0] = sample_num_2;
					  }
				  }
				  if (distance_max_min_.cpu_data()[0] > 0.0001){
					  inter_loss += margin - distance_max_min_.cpu_data()[0];
					  //compute gradient
					  const int sample_1 = static_cast<int>(sample_max_min_1_.cpu_data()[0]);
					  const int sample_2 = static_cast<int>(sample_max_min_2_.cpu_data()[0]);
					  caffe_sub(K_, bottom_data + sample_1*K_, bottom_data + sample_2*K_, dist_inter_.mutable_cpu_data());
					  caffe_axpy(K_, (Dtype)(-2.0) * inter_weight_, dist_inter_.cpu_data(), bout + sample_1*K_);
					  caffe_axpy(K_, (Dtype)(2.0) * inter_weight_, dist_inter_.cpu_data(), bout + sample_2*K_);	
				  }

			  }
		  }
		  //}
	  }
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  inter_loss = std::max(inter_loss, Dtype(0.0));
  //////////////////////////////////////////////////////////////////////////
  Dtype loss(0.0);
  loss = intra_loss*intra_weight_ + inter_loss*inter_weight_;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void IsomapManifoldLearningLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  // Gradient with respect to centers
  /*if (this->param_propagate_down_[0]) {
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
	Dtype* variation_sum_data = variation_sum_data_.mutable_cpu_data();
	const Dtype* center = this->blobs_[0]->cpu_data();

    for (int m = 0; m < M_; m++) {
	    for (int k = 0; k < class_k_; k++){
			int count = 0;
			caffe_set(Data_K_, (Dtype)0., variation_sum_data_.mutable_cpu_data());
		    for (int n = 0; n < N_; n++) {
			    const int label_value = static_cast<int>(label[n]);
				const int sub_class_value = static_cast<int>(class_num_.cpu_data()[n]);
				if (label_value == m && sub_class_value == k) {
			        count++;
					caffe_sub(Data_K_, center + label_value*class_k_*Data_K_ + sub_class_value*Data_K_, bottom[2]->cpu_data() + n*Data_K_, dist_center_.mutable_cpu_data());
					caffe_axpy(Data_K_, (Dtype)(1.0), dist_center_.cpu_data(), variation_sum_data);
			    }
		    }
			caffe_axpy(Data_K_, (Dtype)1. / (count + (Dtype)1.), variation_sum_data_.cpu_data(), center_diff + m * class_k_ * Data_K_ + k * Data_K_);
        }
    }
  }*/
  // Gradient with respect to bottom data
  if (propagate_down[0]) {
	  const Dtype alpha = top[0]->cpu_diff()[0];
	  int num = bottom[0]->num();
	  int channels = bottom[0]->channels();
	  for (int i = 0; i < num; i++){
		  Dtype* bout = bottom[0]->mutable_cpu_diff();
		  caffe_scal(channels, alpha, bout + (i*channels));
	  }
  }
  if (propagate_down[1]) {
	  LOG(FATAL) << this->type()
		  << " Layer cannot backpropagate to label inputs.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(IsomapManifoldLearningLossLayer);
#endif

INSTANTIATE_CLASS(IsomapManifoldLearningLossLayer);
REGISTER_LAYER_CLASS(IsomapManifoldLearningLoss);

}  // namespace caffe