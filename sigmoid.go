package goneurax

import "math"

/**
 * Sigmoid are is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve.
 * Here S(x) is defined by S(x) = 1 / 1 + e^(-x) = e^(x) / e^(x) + 1
 * (Logistic function) -> S(x) = ]0,1[ for x ∈ R
 *
 * It's commonly used function for neural network to normalize the sum of data inputs after applying weights.
 */

// Sigmoid is an empty struct type used as method collection for signoid
type Sigmoid struct{}

func (s *Sigmoid) activate(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (s *Sigmoid) derivate(x float64) float64 {
	sx := s.activate(x)
	return sx * (1 - sx)
}
