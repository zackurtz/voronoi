#include "voronoi.h"



__device__ float euclidean_dis(int2 pix, float2 point) {
	return powf(pix.x - point.x, 2) + powf(pix.y - point.y, 2);
}

__global__ void kBruteForceVoronoi(int* id_image, int2 image_size, float2* point_positions, int number_of_points) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;



	int2 pix_pos = make_int2(gid % image_size.x, gid/image_size.x);

	if (pix_pos.x < image_size.x && pix_pos.y < image_size.y) {
		float min_distance = euclidean_dis(pix_pos, point_positions[0]);
		int min_location = 0;

		for (int i=1; i < number_of_points; ++i) {
			float new_distance = euclidean_dis(pix_pos, point_positions[i]);

			if (new_distance < min_distance) {
				min_distance = new_distance;
				min_location = i;
			}
		}


		id_image[pix_pos.y + pix_pos.x*image_size.y] = min_location;
	}
}



void BruteForceVoronoi(int* id_image, int2 image_dims, float2* point_positions, int num_points) {
	size_t block_size = 256;
	int num_blocks = (image_dims.x*image_dims.y - 1) / block_size + 1;

	kBruteForceVoronoi<<<num_blocks, block_size>>>(id_image, image_dims, point_positions, num_points);
}


__global__ void kIndexToColors(int* index_image, int image_size, uchar4* color_image, uchar4* color_key, int color_key_size) {
	int gid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gid < image_size) {
		int key_idx = index_image[gid];

		if (key_idx < color_key_size) {
			color_image[gid] = color_key[key_idx];
		}
	}
}



void IndexToColors(gpu::array<int>& index_image, gpu::array<uchar4>& color_image, gpu::array<uchar4>& color_key) {
	size_t block_size = 256;
	int num_blocks = (index_image.size() - 1) / block_size + 1;

	kIndexToColors<<<num_blocks, block_size>>>(index_image.raw_pointer(), index_image.size(), color_image.raw_pointer(), color_key.raw_pointer(), color_key.size());

}