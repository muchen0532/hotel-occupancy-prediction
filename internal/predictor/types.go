package predictor

// PredictRequest 单条预测请求
type PredictRequest struct {
	HotelID string   `json:"hotel_id" binding:"required"`
	Date    string   `json:"date" binding:"required"` // YYYY-MM-DD
	Models  []string `json:"models"`                  // ["arima","xgboost","transformer"]，空=全部
}

// BatchPredictRequest 批量预测请求
type BatchPredictRequest struct {
	HotelIDs []string `json:"hotel_ids" binding:"required"`
	Date     string   `json:"date"      binding:"required"`
	Models   []string `json:"models"`
}

// ModelPrediction 单个模型的预测结果
type ModelPrediction struct {
	Model      string  `json:"model"`
	Prediction float64 `json:"prediction"`
	Unit       string  `json:"unit"` // "occupancy_rate_%"
}

// PredictResult 单家酒店预测结果
type PredictResult struct {
	HotelID     string            `json:"hotel_id"`
	Date        string            `json:"date"`
	Predictions []ModelPrediction `json:"predictions"`
	HotelMeta   *HotelMeta        `json:"hotel_meta,omitempty"`
}

// HotelMeta 酒店元数据
type HotelMeta struct {
	HotelID             string `json:"hotel_id"`
	BrandTier           string `json:"brand_tier"`
	HotelDistrict       string `json:"hotel_district"`
	DistrictFunctional  string `json:"district_functional_tier"`
	IsChain             int    `json:"is_chain"`
	VolatilityGroup     string `json:"volatility_group"`
}

// HistoryRecord 历史入住率记录
type HistoryRecord struct {
	HotelID           string  `json:"hotel_id"`
	RecordDate        string  `json:"record_date"`
	OccupancyRate     float64 `json:"occupancy_rate"`
	BrandTier         string  `json:"brand_tier"`
	DistrictFunctional string `json:"district_functional_tier"`
}

// MetricsResponse 模型性能指标
type MetricsResponse struct {
	Summary  []ModelMetrics `json:"summary"`
	TestStart string        `json:"test_start"`
	NHotels  int           `json:"n_hotels"`
}

type ModelMetrics struct {
	Model    string  `json:"model"`
	MAEMean  float64 `json:"MAE_mean"`
	MAEStd   float64 `json:"MAE_std"`
	RMSEMean float64 `json:"RMSE_mean"`
	SMAPEMean float64 `json:"sMAPE_mean"`
	NHotels  int     `json:"n_hotels"`
}
