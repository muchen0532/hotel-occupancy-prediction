package predictor

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

type ScalerData struct {
	Mean     []float64 `json:"mean"`
	Std      []float64 `json:"std"`
	Features []string  `json:"features"`
}

type FeatureMeta struct {
	FeatureOrder  []string                     `json:"feature_order"`
	LabelEncoders map[string]map[string]string `json:"label_encoders"` // col -> idx_str -> class
	LagDays       []int                        `json:"lag_days"`
	RollWindows   []int                        `json:"roll_windows"`
}

type ModelStore struct {
	ModelsDir string
	PythonExe string

	// XGBoost
	XGBScaler   *ScalerData
	XGBFeatMeta *FeatureMeta
	HotelMeta   map[string]*HotelMeta

	// ARIMA
	ARIMADataPath string

	// 历史入住率
	History map[string][]HistoryRecord

	// 评估报告
	EvalReport *EvalReport
}

type EvalReport struct {
	Summary   []ModelMetrics           `json:"summary"`
	PerHotel  []map[string]interface{} `json:"per_hotel"`
	TestStart string                   `json:"test_start"`
	NHotels   int                      `json:"n_hotels"`
}

func Load(modelsDir, pythonExe string) (*ModelStore, error) {
	ms := &ModelStore{
		ModelsDir: modelsDir,
		PythonExe: pythonExe,
	}

	var err error

	ms.XGBFeatMeta, err = loadFeatureMeta(filepath.Join(modelsDir, "xgboost", "feature_meta.json"))
	if err != nil {
		return nil, fmt.Errorf("load xgb feature_meta: %w", err)
	}

	ms.XGBScaler, err = loadScaler(filepath.Join(modelsDir, "xgboost", "scaler.json"))
	if err != nil {
		return nil, fmt.Errorf("load xgb scaler: %w", err)
	}

	ms.HotelMeta, err = loadHotelMeta(filepath.Join(modelsDir, "xgboost", "hotel_meta.json"))
	if err != nil {
		return nil, fmt.Errorf("load hotel_meta: %w", err)
	}

	ms.ARIMADataPath = filepath.Join(modelsDir, "arima", "arima_data.json")
	if _, err := os.Stat(ms.ARIMADataPath); err != nil {
		fmt.Printf("[WARN] arima_data.json not found: %v\n", err)
	}

	ms.History, err = loadHistory(filepath.Join(modelsDir, "history.json"))
	if err != nil {
		fmt.Printf("[WARN] history.json not found, history queries disabled: %v\n", err)
		ms.History = map[string][]HistoryRecord{}
	}

	ms.EvalReport, err = loadEvalReport(filepath.Join(modelsDir, "eval_report.json"))
	if err != nil {
		fmt.Printf("[WARN] eval_report.json not found: %v\n", err)
	}

	return ms, nil
}

func loadScaler(path string) (*ScalerData, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var s ScalerData
	return &s, json.Unmarshal(b, &s)
}

func loadFeatureMeta(path string) (*FeatureMeta, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m FeatureMeta
	return &m, json.Unmarshal(b, &m)
}

func loadHotelMeta(path string) (map[string]*HotelMeta, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var list []HotelMeta
	if err := json.Unmarshal(b, &list); err != nil {
		return nil, err
	}
	m := make(map[string]*HotelMeta, len(list))
	for i := range list {
		m[list[i].HotelID] = &list[i]
	}
	return m, nil
}

func loadHistory(path string) (map[string][]HistoryRecord, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var records []HistoryRecord
	if err := json.Unmarshal(b, &records); err != nil {
		return nil, err
	}
	m := make(map[string][]HistoryRecord)
	for _, r := range records {
		m[r.HotelID] = append(m[r.HotelID], r)
	}
	return m, nil
}

func loadEvalReport(path string) (*EvalReport, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r EvalReport
	return &r, json.Unmarshal(b, &r)
}

func (ms *ModelStore) Encode(col, value string) int {
	if ms.XGBFeatMeta == nil {
		return -1
	}
	enc, ok := ms.XGBFeatMeta.LabelEncoders[col]
	if !ok {
		return -1
	}
	for idxStr, cls := range enc {
		if cls == value {
			if idx, err := strconv.Atoi(idxStr); err == nil {
				return idx
			}
		}
	}
	return -1
}
