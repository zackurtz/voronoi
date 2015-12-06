#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include <cuda_runtime.h>
#include <cuda.h>
#include "gpu_array.h"

struct VoronoiBuffers {
	gpu::array<int> id_image;
	int2 image_dims;
	gpu::array<float2> point_buffer;
	gpu::array<uchar4> color_image;
	gpu::array<uchar4> color_key;

	VoronoiBuffers(int x_size, int y_size, int max_points) {
		image_dims.x = x_size;
		image_dims.y = y_size;

		id_image.resize(image_dims.x*image_dims.y);

		color_image.resize(x_size*y_size);
		point_buffer.resize(max_points);
		color_key.resize(max_points);
	}


	~VoronoiBuffers() {
	}


};




void BruteForceVoronoi(int* id_image, int2 image_dims, float2* point_positions, int num_points);


inline void BuildIndex(cv::Mat& color_image, std::vector<float2>& points, std::vector<uchar4>& color_key) {
	color_key.reserve(points.size());
	for (float2 point : points) {
		cv::Vec3b color = color_image.at<cv::Vec3b>(point.x, point.y);
		uchar4 ucolor;
		ucolor.x = color[0];
		ucolor.y = color[1];
		ucolor.z = color[2];

		color_key.push_back(ucolor);
	}
}

void IndexToColors(gpu::array<int>& index_image, gpu::array<uchar4>& color_image, gpu::array<uchar4>& color_key);