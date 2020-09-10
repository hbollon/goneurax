package main

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
func (p *Perceptron) forwardPass(x []float64) (sum float64) {
	return sig.activate(vectProduct(p.weights, x) + p.bias)
}

// TODO : Add learning rate to gradients

// Calculate and return gradients of Perceptron weights
// wi = wi - (α * θloss/θwi)
func (p *Perceptron) gradWeights(x []float64, y float64) []float64 {
	pred := p.forwardPass(x)
	return vectMatProduct(-(pred-y)*pred*(1-pred), x)
}

// Calculate and return gradient of Perceptron bias
// b = b - (α * θloss/θb)
func (p *Perceptron) gradBias(x []float64, y float64) float64 {
	pred := p.forwardPass(x)
	return -(pred - y) * pred * (1 - pred)
}

// Perceptron training function
func (p *Perceptron) train() {
	for i := 0; i < p.epochs; i++ {
		// Declare variables for futures new weights and bias
		dw := make([]float64, len(p.input[0]))
		db := 0.0
		// Iterate through Perceptron input and calculate their weights and bias gradients using Chain Rule
		for length, val := range p.input {
			dw = vectAdd(dw, p.gradWeights(val, p.actualOutput[length]))
			db += p.gradBias(val, p.actualOutput[length])
		}
		p.weights = vectAdd(p.weights, vectMatProduct(2/float64(len(p.actualOutput)), dw))
		p.bias += db * 2 / float64(len(p.actualOutput))
	}
}
