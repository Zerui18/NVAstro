typedef float q;

void op_assign(q *array, size_t sizeArray, q value);
void op_reduce_average_2d(q *arrays, size_t nArrays, size_t sizeArray);
void op_subtract_pair(q *array1, q *array2, size_t sizeArray);