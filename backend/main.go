package main

import (
	"log"
	"net/http"
	"strings"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	ort "github.com/yalue/onnxruntime_go"
)

// RequestBody
type RequestBody struct {
	PostalImage string `json:"postalImage" binding:"required"`
}

func main() {
	// Initialize ONNX Runtime
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatal(err)
	}
	defer ort.DestroyEnvironment()

	reader, err := NewPostalCodeReader()
	if err != nil {
		panic(err)
	}
	defer reader.Close()

	r := gin.Default()

	r.Use(cors.New(cors.Config{
		AllowOrigins: []string{"http://localhost:4200"},
		AllowMethods: []string{"GET", "POST", "OPTIONS"},
		AllowHeaders: []string{"Origin", "Content-Type", "Accept"},
	}))

	r.POST("/get-postcode", func(c *gin.Context) {
		var req RequestBody

		// Bind and validate the incoming JSON body
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "Invalid request body. 'postalImage' is required.",
			})
			return
		}

		// get base64 sting
		rawBase64 := req.PostalImage
		if idx := strings.Index(rawBase64, ","); idx != -1 {
			rawBase64 = rawBase64[idx+1:]
		}

		postcode, err := reader.Predict(rawBase64)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": "Failed to extract character",
			})
			return
		}
		c.String(http.StatusOK, postcode)
	})

	r.Run(":8080")
}
