#pragma once

double blasLUInvQudaVerify(void *ref_array, void *dev_array_inv, uint64_t array_size, QudaBLASParam *blas_param);

void prepare_ref_array(void *array, int batches, uint64_t array_size, size_t data_size, QudaBLASDataType data_type);

void copy_array(void *array_out, void *array_in, int batches, uint64_t array_size, size_t data_out_size,
                QudaBLASDataType data_type);
