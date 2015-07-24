package vec

import (
	"fmt"
)

type Vector []float64

// Vector dot product.
func Dot(a, b Vector) float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("To take the dot product of two vectors, both vectors must be of the same dimensions. (%v != %v)", len(a), len(b)))
	}
	var ret float64
	for i, _ := range a {
		ret += a[i] * b[i]
	}
	return ret
}

// Element-wise multiply.
func Mul(a, b Vector) Vector {
	if len(a) != len(b) {
		panic(fmt.Sprintf("To take the dot product of two vectors, both vectors must be of the same dimensions. (%v != %v)", len(a), len(b)))
	}
	ret := make(Vector, len(a))
	for i, _ := range a {
		ret[i] = a[i] * b[i]
	}
	return ret
}