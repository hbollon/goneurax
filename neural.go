package goneurax

import (
	"math/rand"
	"time"
)

// Perceptron is a SLP (Single Layout Percepton) representation
type Perceptron struct {
	input        [][]float64 // Input layer
	actualOutput []float64   // Output layer
	weights      []float64
	bias         float64
	epochs       int // Cycles through the full training dataset
}

var sig Sigmoid

// Init function to initialize new Perceptron (or reset existing one)
// Weights of the neural network are set to random float values (between 0..1)
// Bias is set to 0
func (p *Perceptron) init() { //Random Initialization
	rand.Seed(time.Now().UnixNano())
	p.bias = 0.0
	p.weights = make([]float64, len(p.input[0]))
	for i := 0; i < len(p.input[0]); i++ {
		p.weights[i] = rand.Float64()
	}
}

// Forward propagation function
func (p *Perceptron) forwordPass(x []float64) (sum float64) {
	return sig.activate(vectProduct(p.weights, x) + p.bias)
}
