package main

import (
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"

	"gocv.io/x/gocv"

	"github.com/google/uuid"
	ort "github.com/yalue/onnxruntime_go"
)

// Box
type Box struct {
	X, Y, W, H int
}

type TensorBlock struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
}

func NewTensorBlock(
	model string,
	inputName string,
	outputName string,
) (*TensorBlock, error) {
	inputTensor, err := ort.NewEmptyTensor[float32](
		ort.NewShape(1, 28, 28, 1),
	)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	outputTensor, err := ort.NewEmptyTensor[float32](
		ort.NewShape(1, 10),
	)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		model,
		[]string{inputName},
		[]string{outputName},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("load ONNX model: %w", err)
	}
	return &TensorBlock{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}, nil
}

func (r *TensorBlock) Close() error {
	var firstErr error

	if r.session != nil {
		if err := r.session.Destroy(); err != nil {
			firstErr = err
		}
		r.session = nil
	}

	if r.inputTensor != nil {
		if err := r.inputTensor.Destroy(); err != nil && firstErr == nil {
			firstErr = err
		}
		r.inputTensor = nil
	}

	if r.outputTensor != nil {
		if err := r.outputTensor.Destroy(); err != nil && firstErr == nil {
			firstErr = err
		}
		r.outputTensor = nil
	}

	return firstErr
}

type PostalCodeReader struct {
	mu sync.Mutex

	model1 *TensorBlock
	model2 *TensorBlock
	model3 *TensorBlock
}

func NewPostalCodeReader() (*PostalCodeReader, error) {
	inputName := "inputs"
	outputName := "output_0"

	block1, err := NewTensorBlock(
		"model/mnist1.onnx",
		inputName,
		outputName,
	)
	if err != nil {
		return nil, fmt.Errorf("create tensor: %w", err)
	}

	block2, err := NewTensorBlock(
		"model/mnist2.onnx",
		inputName,
		outputName,
	)
	if err != nil {
		return nil, fmt.Errorf("create tensor: %w", err)
	}

	block3, err := NewTensorBlock(
		"model/mnist3.onnx",
		inputName,
		outputName,
	)
	if err != nil {
		return nil, fmt.Errorf("create tensor: %w", err)
	}

	return &PostalCodeReader{
		model1: block1,
		model2: block2,
		model3: block3,
	}, nil
}

func (r *PostalCodeReader) Predict(rawBase64 string) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// decode the base64 to mat
	mat, err := base64ToMat(rawBase64)
	if err != nil {
		return "", err
	}
	defer mat.Close()

	result, err := r.extractCharacters(mat)
	if err != nil {
		return "", err
	}

	return result, nil
}

func (r *PostalCodeReader) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	var firstErr error

	if r.model1 != nil {
		if err := r.model1.Close(); err != nil {
			firstErr = err
		}
		r.model1 = nil
	}
	if r.model2 != nil {
		if err := r.model2.Close(); err != nil {
			firstErr = err
		}
		r.model2 = nil
	}
	if r.model3 != nil {
		if err := r.model3.Close(); err != nil {
			firstErr = err
		}
		r.model3 = nil
	}

	return firstErr
}

func (r *PostalCodeReader) extractCharacters(imageMat gocv.Mat) (string, error) {
	// Change to black
	binary := gocv.NewMat()
	defer binary.Close()

	gocv.Threshold(
		imageMat,
		&binary,
		0,
		255,
		gocv.ThresholdBinaryInv|gocv.ThresholdOtsu,
	)

	// Find contours.
	contours := gocv.FindContours(
		binary,
		gocv.RetrievalExternal,
		gocv.ChainApproxSimple,
	)
	defer contours.Close()

	boxes := make([]Box, 0)

	// Find bounding boxes.
	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		rect := gocv.BoundingRect(contour)
		x := rect.Min.X
		y := rect.Min.Y
		w := rect.Dx()
		h := rect.Dy()

		if w > 20 && h > 20 {
			boxes = append(boxes, Box{
				X: x,
				Y: y,
				W: w,
				H: h,
			})
		}
	}

	// Sort by vertical position and horizontal position
	sort.Slice(boxes, func(i, j int) bool {
		if boxes[i].Y != boxes[j].Y {
			return boxes[i].Y < boxes[j].Y
		}
		return boxes[i].X < boxes[j].X
	})

	// Crop each character and predict
	result := ""
	id, err := uuid.NewV7()
	if err == nil {
		os.MkdirAll(fmt.Sprintf("history/%s", id), os.ModePerm)
		for index, box := range boxes {
			x1 := box.X + 5
			y1 := box.Y + 5
			x2 := box.X + box.W - 5
			y2 := box.Y + box.H - 5

			if x2 <= x1 || y2 <= y1 {
				continue
			}

			// get character
			region := binary.Region(image.Rect(x1, y1, x2, y2))
			character := region.Clone()
			digit := r.predictDigit(character, index, id.String())
			region.Close()
			character.Closed()

			result += strconv.Itoa(digit)
		}
	}

	return result, nil
}

func (r *PostalCodeReader) predictDigit(img gocv.Mat, position int, folder string) int {
	// make image padding
	h := img.Rows()
	w := img.Cols()
	borderX := int(float64(w) * 0.80)
	borderY := int(float64(h) * 0.40)
	padded := gocv.NewMat()
	defer padded.Close()
	gocv.CopyMakeBorder(
		img,
		&padded,
		borderY,
		borderY,
		borderX,
		borderX,
		gocv.BorderConstant,
		color.RGBA{R: 0, G: 0, B: 0, A: 0},
	)

	// Resize to 28x28
	resized := gocv.NewMat()
	defer resized.Close()
	gocv.Resize(
		padded,
		&resized,
		image.Pt(28, 28),
		0,
		0,
		gocv.InterpolationLinear,
	)

	// Convert image to float32 and normalize
	inputData := make([]float32, 28*28)
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			pixel := resized.GetUCharAt(y, x)
			inputData[y*28+x] = float32(pixel) / 255.0
		}
	}

	probabilities1 := r.getModelProbabilities(1, inputData)
	probabilities2 := r.getModelProbabilities(2, inputData)
	probabilities3 := r.getModelProbabilities(3, inputData)
	probabilities := ensemble(probabilities1, probabilities2, probabilities3)
	prediction := argmax(probabilities)

	gocv.IMWrite(fmt.Sprintf("history/%s/%d-%d.png", folder, position, prediction), padded)
	return prediction
}

func (r *PostalCodeReader) getModelProbabilities(model int, inputData []float32) []float32 {
	inputTensor, err := ort.NewTensor(
		ort.NewShape(1, 28, 28, 1),
		inputData,
	)
	if err != nil {
		log.Fatal(err)
	}
	defer inputTensor.Destroy()

	tensor := r.model1
	if model == 2 {
		tensor = r.model2
	} else if model == 3 {
		tensor = r.model3
	}

	src := inputTensor.GetData()
	dst := tensor.inputTensor.GetData()

	copy(dst, src)

	// Predict
	if err := tensor.session.Run(); err != nil {
		log.Fatal(err)
	}
	return tensor.outputTensor.GetData()
}

func base64ToMat(base64Str string) (gocv.Mat, error) {
	// Handle Base64 strings with a data URI prefix
	if idx := strings.Index(base64Str, ","); strings.HasPrefix(base64Str, "data:") && idx >= 0 {
		base64Str = base64Str[idx+1:]
	}

	// Decode Base64
	data, err := base64.StdEncoding.DecodeString(base64Str)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("base64 decode failed: %w", err)
	}

	// Decode image bytes into Mat
	mat, err := gocv.IMDecode(data, gocv.IMReadGrayScale)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("image decode failed: %w", err)
	}

	if mat.Empty() {
		mat.Close()
		return gocv.NewMat(), fmt.Errorf("decoded image is empty")
	}

	return mat, nil
}

func ensemble(probs1, probs2, probs3 []float32) []float32 {
	result := make([]float32, len(probs1))

	for i := range result {
		result[i] = (probs1[i] + probs2[i] + probs3[i]) / 3.0
	}

	return result
}

func argmax(values []float32) int {
	index := 0

	for i := 1; i < len(values); i++ {
		if values[i] > values[index] {
			index = i
		}
	}

	return index
}
