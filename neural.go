package goneurax

// Perceptron is a SLP (Single Layout Percepton) representation
type Perceptron struct {
	input        [][]float64 // Input layer
	actualOutput []float64   // Output layer
	weights      []float64
	bias         float64
	epochs       int // Cycles through the full training dataset
}
