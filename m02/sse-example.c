#include <immintrin.h>  // Include SSE4 intrinsics header
#include <stdio.h>

// Function to perform SSE4-based vector matrix multiplication
void vectorMatrixMultiply(const float* matrix, const float* inputVector,
                          float* resultVector) {
  __m128 row1 = _mm_loadu_ps(matrix);        // Load the first row of the matrix
  __m128 row2 = _mm_loadu_ps(matrix + 4);    // Load the second row
  __m128 row3 = _mm_loadu_ps(matrix + 8);    // Load the third row
  __m128 row4 = _mm_loadu_ps(matrix + 12);   // Load the fourth row
  __m128 input = _mm_loadu_ps(inputVector);  // Load the input vector

  // Perform the vector matrix multiplication
  // Add all four element-wise products together
  __m128 result =
      _mm_add_ps(_mm_add_ps(_mm_mul_ps(row1, input), _mm_mul_ps(row2, input)),
                 _mm_add_ps(_mm_mul_ps(row3, input), _mm_mul_ps(row4, input)));

  // Store the result back into memory
  _mm_storeu_ps(resultVector, result);
}

int main() {
  float matrix[16] = {1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                      9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
  float inputVector[4] = {2.0, 3.0, 4.0, 1.0};
  float resultVector[4];

  vectorMatrixMultiply(matrix, inputVector, resultVector);

  printf("Result: %f %f %f %f\n", resultVector[0], resultVector[1],
         resultVector[2], resultVector[3]);

  return 0;
}