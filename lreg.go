package lreg

import "fmt"
import "sync"

import (
	"github.com/sdleffler/goml/vec"
)

type Predictor (func (vec.Vector) vec.Vector) // Hypothesis function h(x).

type State struct{
	dim uint // Dimensionality of the dataset.
	theta vec.Vector // Theta vector coefficients for state update.
	alpha float64 // Alpha.
	samples []Sample // Sample points. Always one dimension larger than 
	predictor Predictor
}

type Sample struct{
	x vec.Vector // "x" values
	y float64 // Output
}

func MakeSample(vert []float64) Sample {
	if len(vert) > 0 {
		x := make([]float64, len(vert) - 1)
		copy(x, vert)
		y := vert[len(vert) - 1]
		return Sample{
			x, y
		}
	} else {
		return Sample{
			[], 0
		}
	}
}

func NewState(s_dim uint, sams []Sample) *State {
	for _, v := range sams {
		if s_dim != len(v.x) { panic(fmt.Sprint("Samples must be of dimensionality ", s_dim, "!")) }
	}
	return &State{ s_dim, make(vec.Vector, s_dim), 0.01, sams }
}

func (st *State) StepBatch() {
	theta_next := make(vec.Vector, len(st.theta))
	errs := make([]float64, len(st.samples))
	for i, sample := range st.samples {
		errs[i] = (sample.y - vec.Dot(st.theta, st.predictor(sample.x)))
	}
	for j, theta_j := range theta {
		var sum float64
		for i, err := range errs {
			sum += st.samples[i][j] * err
		}
		sum *= st.alpha
		theta_next[j] = theta_j - sum
	}
	st.theta = theta_next
}