package main

/**
 * Vector operations functions
 */

// Addition of two vector (of the same size)
func vectAdd(v1, v2 []float64) []float64 {
	add := make([]float64, len(v1))
	for i := 0; i < len(v1); i++ {
		add[i] = v1[i] + v2[i]
	}
	return add
}

// Product of two vector (of the same size)
func vectProduct(v1, v2 []float64) float64 {
	var product float64
	for i := 0; i < len(v1); i++ {
		product += v1[i] * v2[i]
	}
	return product
}

// Scalar product between vector and matrix
func vectMatProduct(s float64, mat []float64) []float64 {
	result := make([]float64, len(mat))
	for i := 0; i < len(mat); i++ {
		result[i] += s * mat[i]
	}
	return result
}
