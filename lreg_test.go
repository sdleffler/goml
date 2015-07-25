package goml

import "testing"
import "fmt"
import "math"

type mapFunc func ([]float64) float64

func MakeDataSet(f mapFunc, x_range [][2]float64, n_sams uint) []Sample {
	st_size := make([]float64, len(x_range))
	for i, r := range x_range {
		st_size[i] = (r[1] - r[0]) / float64(n_sams)
	}
	d_vtxs := populate(x_range, st_size, n_sams)
	for i, x := range d_vtxs {
		d_vtxs[i] = append(x, f(x))
	}
	//fmt.Println("Populated data set, ", len(d_vtxs), " points: ", d_vtxs)
	return MakeSampleSet(d_vtxs)
}

func populate(x_range [][2]float64, st_size []float64, n_sams uint) [][]float64 {
	x_i := make([]float64, n_sams)
	for i, _ := range x_i {
		x_i[i] = x_range[0][0] + (st_size[0] * float64(i))
	}
	if (len(x_range) > 1) {
		x_j := populate(x_range[1:], st_size[1:], n_sams)
		x := [][]float64{}
		for i, _ := range x_i {
			n_x_j := make([][]float64, len(x_j))
			for j, _ := range x_j {
				n_x_j[j] = append([]float64{ x_i[i] }, x_j[j]...)
			}
			x = append(x, n_x_j...)
		}
		return x
	} else {
		x_i_wrap := make([][]float64, len(x_i))
		for i, _ := range x_i_wrap {
			x_i_wrap[i] = []float64{ x_i[i] }
		}
		return x_i_wrap
	}
}

func TestLinreg(t *testing.T) {
	d_sms := MakeDataSet(
		func (x []float64) float64 {
			fmt.Println(x)
			return 1 + 2*x[0] + 0.5*math.Pow(x[0], 2)
		},
		[][2]float64{ 
			[2]float64{-7, 7},
		},
		1000)
	d_pred := func (vec Vector) Vector {
		return Vector{ 1, vec[0], vec[0]*vec[0] }
	}
	lr_state := NewState(1e-4, 1, d_sms, 3, d_pred)
	fmt.Println("[ 0 ] Theta: ", lr_state.theta)
	for i := 1; i < 1500; i++ {
		lr_state.StepStochastic()
		fmt.Println("[", i, "] Theta: ", lr_state.theta)
	}
	fmt.Println("[ final ] Theta: ", lr_state.theta)
}