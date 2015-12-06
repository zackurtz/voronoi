#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <vector>

namespace gpu {

template<typename T>
T* allocate(size_t array_size) {
	T* temp;
	cudaMalloc(&temp, array_size*sizeof(T));

	return temp;
}

template<typename T>
void free(T* array_pointer) {
	if (array_pointer) {
		cudaFree(array_pointer);
	}
}

template<typename T>
class array {
public:
	array() {
		memory_ = nullptr;
		length_ = 0;
		owned_ = true;
	}

	array(size_t initial_length) {
		resize(initial_length);
		owned_ = true;
	}

	array(const array& other) {
		memory_ = other.memory_;
		length_ = other.length;
		owned_ = false;
	}

	~array() {
		if (owned_ && memory_) {
			free(memory_);
		}
	}

	void resize(size_t length) {
		assert(owned_);

		if (memory_ && owned_) {
			gpu::free(memory_);
		}
		memory_ = allocate<T>(length);
		length_ = length;
	}

	void upload(const std::vector<T>& src) {
		if (!memory_) {
			throw std::runtime_error("GPU memory not yet allocated");
		}
		assert(src.size() == length_);

		cudaMemcpy(memory_, &src[0], sizeof(T)*src.size(), cudaMemcpyHostToDevice);
	}

	void download(std::vector<T>& dest) {
		if (!memory_) {
			throw std::runtime_error("GPU memory not yet allocated");
		}
		if(dest.size() != length_) {
			dest.resize(length_);
		}

		cudaMemcpy(&dest[0], memory_,  sizeof(T)*dest.size(), cudaMemcpyDeviceToHost);
	}

	void download(T* dest) {
		if (!memory_) {
			throw std::runtime_error("GPU memory not yet allocated");
		}
		cudaMemcpy(dest, memory_,  sizeof(T)*size(), cudaMemcpyDeviceToHost);		
	}

	T* raw_pointer() { return memory_; }
	size_t size() { return length_; }

private:
	T* memory_;
	size_t length_;
	bool owned_;
};





}