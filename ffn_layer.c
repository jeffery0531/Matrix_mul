#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

void linear_layer( size_t seq_len, size_t input_dim, size_t output_dim, 
                  float* input, float* weights, float* output) {
   
   size_t j,k,i,kk,jj;
   int b_row =16;
   int b_col = 256;

   #pragma GCC unroll 32
   for( kk = 0;kk<input_dim;kk+=b_row){
     
     for( jj = 0;jj<output_dim;jj+=b_col){
       #pragma GCC unroll 2
       for( i=0;i<seq_len;i+=4){
          #pragma GCC unroll 2 
         for( j = jj;j<jj + b_col;j+=16){
            
	         //float sum = output[i*output_dim + j];
           __m256 sum = _mm256_load_ps(&output[i*output_dim + j]); 
           __m256 sum1 = _mm256_load_ps(&output[(i+1)*output_dim + j]); 

           __m256 sum2 = _mm256_load_ps(&output[(i+2)*output_dim + j]); 
           __m256 sum3 = _mm256_load_ps(&output[(i+3)*output_dim + j]);

           __m256 sum4 = _mm256_load_ps(&output[(i)*output_dim + (j+8)]); 
           __m256 sum5 = _mm256_load_ps(&output[(i+1)*output_dim + (j+8)]); 
           __m256 sum6 = _mm256_load_ps(&output[(i+2)*output_dim + (j+8)]); 
           __m256 sum7 = _mm256_load_ps(&output[(i+3)*output_dim + (j+8)]); 
	         #pragma GCC unroll 16
	         for( k =kk;k<(kk+b_row);k++){
               
	     //_mm_prefetch((float*)&weights[output_dim*(k+1)+j], _MM_HINT_T1);
                    
             __m256 input_val = _mm256_set1_ps(input[i * input_dim + k]);
             __m256 input_val1 = _mm256_set1_ps(input[(i+1) * input_dim + k]);
             __m256 input_val2 = _mm256_set1_ps(input[(i+2) * input_dim + k]);
             __m256 input_val3 = _mm256_set1_ps(input[(i+3) * input_dim + k]);

             __m256 weights_vec = _mm256_load_ps(&weights[output_dim * k + j]);
             __m256 weights_vec1 = _mm256_load_ps(&weights[output_dim * k + (j+8)]);

             sum = _mm256_fmadd_ps(input_val, weights_vec, sum);
             sum1 = _mm256_fmadd_ps(input_val1, weights_vec, sum1);
             sum2 = _mm256_fmadd_ps(input_val2, weights_vec, sum2);
             sum3 = _mm256_fmadd_ps(input_val3, weights_vec, sum3);
             sum4 = _mm256_fmadd_ps(input_val, weights_vec1, sum4);
             sum5 = _mm256_fmadd_ps(input_val1, weights_vec1, sum5);
             sum6 = _mm256_fmadd_ps(input_val2, weights_vec1, sum6);
             sum7 = _mm256_fmadd_ps(input_val3, weights_vec1, sum7);

	     
	   }
                      
	 _mm256_store_ps(&output[i * output_dim + j], sum);
     _mm256_store_ps(&output[(i+1) * output_dim + j], sum1);
     _mm256_store_ps(&output[(i+2) * output_dim + j], sum2);
     _mm256_store_ps(&output[(i+3) * output_dim + j], sum3);

     	 _mm256_store_ps(&output[(i) * output_dim + j+8], sum4);
     _mm256_store_ps(&output[(i+1)  * output_dim + j+8], sum5);
     _mm256_store_ps(&output[(i+2)  * output_dim + j+8], sum6);
     _mm256_store_ps(&output[(i+3)  * output_dim + j+8], sum7);

   }
       
       }
     }
   }   

}

void ffn_layer( size_t seq_len, size_t input_dim, size_t hidden_dim, size_t output_dim, 
               float* x, float* up_proj_weight, float* gate_proj_weight, float* down_proj_weight, 
               float* output,float* up_proj_result, float* gate_proj_result) {

    
    linear_layer( seq_len, input_dim, hidden_dim, x, up_proj_weight, up_proj_result);
  
    linear_layer( seq_len, input_dim, hidden_dim, x, gate_proj_weight, gate_proj_result);

    size_t data = seq_len * hidden_dim;
    size_t i = 0;

    #pragma GCC unroll 32
    for (; i + 7 < data; i += 8) {
        __m256 gate_vec = _mm256_load_ps(&gate_proj_result[i]);
        __m256 up_vec = _mm256_load_ps(&up_proj_result[i]);
        
        __m256 zero_vec = _mm256_setzero_ps(); 
        gate_vec = _mm256_max_ps(gate_vec, zero_vec);

        __m256 result_vec = _mm256_mul_ps(up_vec, gate_vec);

        _mm256_store_ps(&gate_proj_result[i], gate_vec);
        _mm256_store_ps(&up_proj_result[i], result_vec);
    }


    linear_layer( seq_len, hidden_dim, output_dim, up_proj_result, down_proj_weight, output);


}
