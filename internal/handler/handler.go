package handler

import (
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"hotel-occupancy-prediction/internal/predictor"
)

type Handler struct {
	Store *predictor.ModelStore
}

func New(store *predictor.ModelStore) *Handler {
	return &Handler{Store: store}
}

func (h *Handler) Predict(c *gin.Context) {
	var req predictor.PredictRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	date, err := time.Parse("2006-01-02", req.Date)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "date format must be YYYY-MM-DD"})
		return
	}
	result, err := h.Store.Predict(req.HotelID, date, req.Models)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, result)
}

func (h *Handler) History(c *gin.Context) {
	hotelID := c.Query("hotel_id")
	if hotelID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "hotel_id required"})
		return
	}
	records, ok := h.Store.History[hotelID]
	if !ok {
		c.JSON(http.StatusNotFound, gin.H{"error": "hotel_id not found"})
		return
	}
	start, end := c.Query("start"), c.Query("end")
	filtered := records
	if start != "" || end != "" {
		filtered = []predictor.HistoryRecord{}
		for _, r := range records {
			if start != "" && r.RecordDate < start {
				continue
			}
			if end != "" && r.RecordDate > end {
				continue
			}
			filtered = append(filtered, r)
		}
	}
	c.JSON(http.StatusOK, gin.H{
		"hotel_id":   hotelID,
		"hotel_meta": h.Store.HotelMeta[hotelID],
		"records":    filtered,
		"total":      len(filtered),
	})
}

func (h *Handler) Metrics(c *gin.Context) {
	if h.Store.EvalReport == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "eval report not loaded"})
		return
	}
	modelFilter := strings.ToLower(c.Query("model"))
	summary := h.Store.EvalReport.Summary
	if modelFilter != "" {
		var filtered []predictor.ModelMetrics
		for _, m := range summary {
			if strings.Contains(strings.ToLower(m.Model), modelFilter) {
				filtered = append(filtered, m)
			}
		}
		summary = filtered
	}
	c.JSON(http.StatusOK, predictor.MetricsResponse{
		Summary:   summary,
		TestStart: h.Store.EvalReport.TestStart,
		NHotels:   h.Store.EvalReport.NHotels,
	})
}

func (h *Handler) Hotels(c *gin.Context) {
	list := make([]*predictor.HotelMeta, 0, len(h.Store.HotelMeta))
	for _, m := range h.Store.HotelMeta {
		list = append(list, m)
	}
	c.JSON(http.StatusOK, gin.H{"total": len(list), "hotels": list})
}

func (h *Handler) Health(c *gin.Context) {
	available := []string{"arima", "xgboost"}

	c.JSON(http.StatusOK, gin.H{
		"status":           "ok",
		"n_hotels":         len(h.Store.HotelMeta),
		"available_models": available,
		"xgb_meta":         h.Store.XGBFeatMeta != nil,
		"history_loaded":   len(h.Store.History) > 0,
		"eval_report":      h.Store.EvalReport != nil,
	})
}
