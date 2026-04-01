package main

import (
	"flag"
	"log"
	"net/http"
	"path/filepath"
	"runtime"

	"github.com/gin-gonic/gin"
	"hotel-occupancy-prediction/internal/handler"
	"hotel-occupancy-prediction/internal/predictor"
)

func main() {
	var (
		port      = flag.String("port", "8080", "HTTP port")
		modelsDir = flag.String("models", "", "Path to models directory")
		pythonExe = flag.String("python", "python", "Python executable path")
		dev       = flag.Bool("dev", false, "Development mode (gin debug)")
	)
	flag.Parse()

	if *modelsDir == "" {
		_, filename, _, _ := runtime.Caller(0)
		*modelsDir = filepath.Join(filepath.Dir(filename), "..", "..", "models")
	}

	if !*dev {
		gin.SetMode(gin.ReleaseMode)
	}

	log.Printf("Loading artifacts from: %s", *modelsDir)
	log.Printf("Python executable: %s", *pythonExe)

	store, err := predictor.Load(*modelsDir, *pythonExe)
	if err != nil {
		log.Fatalf("Failed to load model artifacts: %v", err)
	}
	log.Printf("Loaded: %d hotels | XGB meta=%v | History keys=%d",
		len(store.HotelMeta),
		store.XGBFeatMeta != nil,
		len(store.History),
	)

	h := gin.Default()

	// CORS
	h.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Content-Type")
		if c.Request.Method == http.MethodOptions {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	})

	// 前端页面
	h.StaticFile("/", "./index.html")

	hdl := handler.New(store)
	api := h.Group("/api")
	{
		api.GET("/health", hdl.Health)
		api.GET("/hotels", hdl.Hotels)
		api.GET("/history", hdl.History)
		api.GET("/metrics", hdl.Metrics)
		api.POST("/predict", hdl.Predict)
	}

	log.Printf("Server listening on :%s  →  http://localhost:%s", *port, *port)
	if err := h.Run(":" + *port); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}
