package main

import "fmt"

func main() {
	p := Perceptron{
		input:        [][]float64{{0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 0}}, //Input Data
		actualOutput: []float64{0, 1, 1, 0},                                   //Actual Output
		epochs:       1000,                                                    //Number of Epoch
	}

	p.init()
	p.train()

	fmt.Println(p.forwardPass([]float64{0, 1, 0}))
	fmt.Println(p.forwardPass([]float64{1, 0, 1}))
}
