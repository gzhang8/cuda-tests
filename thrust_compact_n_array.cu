#include <iostream>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>
#include <thrust/sequence.h>

typedef thrust::tuple<int, int, int, bool> XYZFlag;

struct should_remove {
  __host__ __device__
  bool operator() (const XYZFlag& tup) {
    const bool flag = thrust::get<3>(tup);
    return !flag;
  }
};

int main() {
  const int N = 4000000;
  int* x_raw_ptr;
  cudaMalloc(&x_raw_ptr, N * sizeof(int));
  int* y_raw_ptr;
  cudaMalloc(&y_raw_ptr, N * sizeof(int));
  int* z_raw_ptr;
  cudaMalloc(&z_raw_ptr, N * sizeof(int));
  bool* should_keep_raw;
  cudaMalloc(&should_keep_raw, N * sizeof(bool));

  // bind device_ptr
  thrust::device_ptr<int> x_dev_ptr(x_raw_ptr);
  thrust::device_ptr<int> y_dev_ptr(y_raw_ptr);
  thrust::device_ptr<int> z_dev_ptr(z_raw_ptr);
  thrust::device_ptr<bool> should_keep_dev_ptr(should_keep_raw); 

  // init value in dev_ptr s for test
  thrust::sequence(x_dev_ptr, x_dev_ptr + N);
  thrust::sequence(y_dev_ptr, y_dev_ptr + N, N);
  thrust::sequence(z_dev_ptr, z_dev_ptr + N, N*2);
  thrust::fill(should_keep_dev_ptr, should_keep_dev_ptr+N, false);

  //for (int i = 0; i < N; i++) {
  for (int i = 0; i < 10240; i++) {
    //x_dev_ptr[i] = i;
    //y_dev_ptr[i] = i + N;
    //z_dev_ptr[i] = i + N * 2;
    should_keep_dev_ptr[i] = (bool)(i % 2);
  }

  // remove if

  auto first = thrust::make_zip_iterator(thrust::make_tuple(x_dev_ptr, y_dev_ptr, z_dev_ptr, should_keep_dev_ptr));
  auto last = thrust::make_zip_iterator(thrust::make_tuple(x_dev_ptr+N, y_dev_ptr+N,z_dev_ptr+N, should_keep_dev_ptr+N));

  auto newEnd = thrust::remove_if(first, last, should_remove());

  // print result

  for (int i = 0; i < 2; i++) {
    std::cout << "x: " << x_dev_ptr[i];
    std::cout << ", y: " << y_dev_ptr[i];
    std::cout << ", z: " << z_dev_ptr[i];
    std::cout << ", flag: " << should_keep_dev_ptr[i] << std::endl;
  }


  return 0;
}