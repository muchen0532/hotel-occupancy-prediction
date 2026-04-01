package predictor

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

type PredictionResult struct {
	HotelID    string  `json:"hotel_id"`
	Date       string  `json:"date"`
	Prediction float64 `json:"prediction"`
	Error      string  `json:"error"`
}

func (ms *ModelStore) PredictXGB(hotelID, date string, fi *FeatureInput) (float64, error) {
	vec, err := json.Marshal(fi.ToVector(ms.XGBFeatMeta.FeatureOrder))
	if err != nil {
		return 0, err
	}

	scriptPath := filepath.Join(ms.ModelsDir, "..", "python", "predict_xgboost.py")
	cmd := exec.Command(ms.PythonExe, scriptPath, hotelID, date)
	cmd.Env = append(os.Environ(), "FEATURE_VEC="+string(vec))

	out, err := cmd.CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("xgboost subprocess error:\n%s", string(out))
	}

	var res PredictionResult
	if err := json.Unmarshal(out, &res); err != nil {
		return 0, fmt.Errorf("xgboost parse: %w\nraw=%s", err, string(out))
	}
	if res.Error != "" {
		return 0, fmt.Errorf("xgboost: %s", res.Error)
	}
	return clamp(res.Prediction, 0, 100), nil
}

type arimaResult struct {
	HotelID     string    `json:"hotel_id"`
	Steps       int       `json:"steps"`
	Predictions []float64 `json:"predictions"`
	Error       string    `json:"error"`
}

func (ms *ModelStore) PredictARIMA(hotelID string, targetDate time.Time, history []HistoryRecord) (float64, error) {
	steps := 1
	if len(history) > 0 {
		lastDate, err := time.Parse("2006-01-02", history[len(history)-1].RecordDate)
		if err == nil {
			if diff := int(targetDate.Sub(lastDate).Hours() / 24); diff > 0 {
				steps = diff
			}
		}
	}

	scriptPath := filepath.Join(ms.ModelsDir, "..", "python", "predict_arima.py")
	cmd := exec.Command(ms.PythonExe, scriptPath, hotelID, fmt.Sprintf("%d", steps))
	out, err := cmd.CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("arima subprocess error:\n%s", string(out))
	}

	var res arimaResult
	if err := json.Unmarshal(out, &res); err != nil {
		return 0, fmt.Errorf("arima parse: %w\nraw=%s", err, string(out))
	}
	if res.Error != "" {
		return 0, fmt.Errorf("arima: %s", res.Error)
	}
	if len(res.Predictions) == 0 {
		return 0, fmt.Errorf("arima: empty predictions")
	}
	return clamp(res.Predictions[len(res.Predictions)-1], 0, 100), nil
}

func (ms *ModelStore) Predict(
	hotelID string,
	date time.Time,
	models []string,
) (*PredictResult, error) {
	meta, ok := ms.HotelMeta[hotelID]
	if !ok {
		return nil, fmt.Errorf("hotel_id %q not found", hotelID)
	}

	history := ms.History[hotelID]
	fi := BuildFeatureInput(hotelID, date, meta, history, ms)
	dateStr := date.Format("2006-01-02")

	if len(models) == 0 {
		models = []string{"arima", "xgboost"}
	}

	result := &PredictResult{
		HotelID:   hotelID,
		Date:      dateStr,
		HotelMeta: meta,
	}

	for _, m := range models {
		var pred float64
		var err error

		switch m {
		case "xgboost":
			pred, err = ms.PredictXGB(hotelID, dateStr, fi)
		case "arima":
			pred, err = ms.PredictARIMA(hotelID, date, history)
		default:
			err = fmt.Errorf("unknown model: %s (available: arima, xgboost, transformer)", m)
		}

		if err != nil {
			result.Predictions = append(result.Predictions, ModelPrediction{
				Model:      m,
				Prediction: -1,
				Unit:       "error: " + err.Error(),
			})
		} else {
			result.Predictions = append(result.Predictions, ModelPrediction{
				Model:      m,
				Prediction: math.Round(pred*100) / 100,
				Unit:       "occupancy_rate_%",
			})
		}
	}
	return result, nil
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
