#ifndef CAFFE_ISOMAP_MANIFOLD_LEARNING_LOSS_LAYER_HPP_
#define CAFFE_ISOMAP_MANIFOLD_LEARNING_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

	template <typename Dtype>
	class IsomapManifoldLearningLossLayer : public LossLayer<Dtype> {
	public:
		explicit IsomapManifoldLearningLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline int ExactNumBottomBlobs() const { return 2 ; }
		virtual inline const char* type() const { return "IsomapManifoldLearningLoss"; }
		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return bottom_index != 1;
		}

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		// virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		//   const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		//virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		//  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


		Dtype inter_weight_, intra_weight_;
		int num_output_, class_k_;
		Blob<Dtype> class_num_; //sub-class of different samples
		Blob<Dtype> distance_sample_k_; 
		//Blob<Dtype> dist_class_;
		Blob<Dtype> dist_intra_;
		Blob<Dtype> dist_inter_;
		Blob<Dtype> distance_inter_;
		Blob<Dtype> distance_inter_1_;
		Blob<Dtype> distance_max_min_;
		Blob<Dtype> distance_max_min_1_;
		Blob<Dtype> sample_max_min_2_;
		Blob<Dtype> sample_max_min_1_;
		Blob<Dtype> sequence_sample_;
		Blob<Dtype> dot_;
		Blob<Dtype> mend_;
		Blob<Dtype> ones_;
		Blob<Dtype> dist_sq_;
		//Blob<Dtype> variation_sum_data_;
		Blob<Dtype> sub_class_num_;
		Blob<Dtype> intra_loss_;
		//Blob<Dtype> dist_inter_1_;
		Blob<Dtype> sum_sub_class_num_;
		Blob<Dtype> data_;
		//Blob<Dtype> dist_center_;
		int Data_K_;
		int M_;
		int N_;
		int K_;	
	};

}  // namespace caffe

#endif  // CAFFE_ISOMAP_MANIFOLD_LEARNING_LOSS_LAYER_HPP_