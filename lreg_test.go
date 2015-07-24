package goml

import "testing"
import "fmt"

func MakeDataSet(f func (float64) float64, x_min float64, x_max float64, n_sams uint) []Sample {
	st_size := (x_max - x_min) / float64(n_sams)
	d_sams := make([]Sample, n_sams)
	for i, _ := range d_sams {
		x := x_min + (st_size * float64(i))
		d_sams[i] = MakeSample([]float64{ x, f(x) })
	}
	return d_sams
}

func TestLinreg(t *testing.T) {
	d_dim := uint(1)
	d_sms := MakeDataSet(func (x float64) float64 {
		return x * x * x
		}, -2, 2, 100)
	d_pred := func (vec Vector) Vector {
		return vec
	}
	lr_state := NewState(d_dim, d_sms, d_pred)
	fmt.Println("[ 0 ] Theta: ", lr_state.theta)
	for i := 1; i < 100; i++ {
		lr_state.StepBatch()
		fmt.Println("[", i, "] Theta: ", lr_state.theta)
	}
}