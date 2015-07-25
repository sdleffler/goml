package goml

import "fmt"

type Predictor (func (Vector) Vector) // Hypothesis function h(x).

type State struct{
	dim uint // Dimensionality of the dataset.
	theta Vector // Theta vector coefficients for state update.
	alpha float64 // Alpha.
	samples []Sample // Sample points. Always one dimension larger than 
	predictor Predictor
}

type Sample struct{
	x Vector // "x" values
	y float64 // Output
}

func MakeSample(vert []float64) Sample {
	if len(vert) > 0 {
		x := make([]float64, len(vert) - 1)
		copy(x, vert)
		y := vert[len(vert) - 1]
		return Sample{ x, y }
	} else {
		return Sample{ make(Vector, 0), 0 }
	}
}

func MakeSampleSet(vert [][]float64) []Sample {
	d_sms := make([]Sample, len(vert))
	for i, v := range vert {
		d_sms[i] = MakeSample(v)
	}
	return d_sms
}

func NewState(alpha float64, s_dim uint, sams []Sample, p_dim uint, pred Predictor) *State {
	for _, v := range sams {
		if int(s_dim) != len(v.x) { panic(fmt.Sprint("Samples must be of dimensionality ", s_dim, "! Did you forget to set the sample dimension?")) }
	}
	return &State{ s_dim, make(Vector, p_dim), alpha, sams, pred }
}

func (st *State) StepBatch() {
	theta_next := make(Vector, len(st.theta))
	errs := make([]float64, len(st.samples))
	for i, sample := range st.samples {
		errs[i] = (sample.y - Dot(st.predictor(sample.x), st.theta))
	}
	for j, theta_j := range st.theta {
		var sum float64
		for i, err := range errs {
			sum += st.predictor(st.samples[i].x)[j] * err
		}
		theta_next[j] = theta_j + st.alpha * sum
	}
	st.theta = theta_next
}

func (st *State) StepStochastic() {
	for j, _ := range st.theta {
		for _, sample := range st.samples {
			st.theta[j] += st.alpha * st.predictor(sample.x)[j] * (sample.y - Dot(st.predictor(sample.x), st.theta))
		}
	}
}

func (st *State) GetPredictor() (func (Vector) float64) {
	return func (vec Vector) float64 {
		return Dot(st.predictor(vec), st.theta)
	}
}