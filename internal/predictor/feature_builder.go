package predictor

import (
	"fmt"
	"math"
	"sort"
	"time"
)

type FeatureInput struct {
	// hotel_meta
	HotelIDEnc  int
	BrandEnc    int
	DistrictEnc int
	FuncTierEnc int
	IsChain     int

	// date 解析
	DayOfWeek  int
	Month      int
	DayOfMonth int
	Quarter    int
	IsWeekend  int
	MonthSin   float64
	MonthCos   float64
	DowSin     float64
	DowCos     float64

	// 日历
	IsPublicHoliday  int
	IsWorkday        int
	IsSchoolVacation int

	// 天气
	Tavg float64
	Tmin float64
	Tmax float64
	Prcp float64
	Snow float64
	Wdir float64
	Wspd float64
	Wpgt float64
	Pres float64
	Tsun float64

	// Lag / Rolling
	Lags        map[int]float64
	RollingMean map[int]float64
	RollingStd  map[int]float64
	RollingMax  map[int]float64
	RollingMin  map[int]float64
}

func BuildFeatureInput(
	hid string,
	date time.Time,
	meta *HotelMeta,
	history []HistoryRecord,
	ms *ModelStore,
) *FeatureInput {
	fi := &FeatureInput{}

	fi.HotelIDEnc = ms.Encode("hotel_id", hid)
	fi.BrandEnc = ms.Encode("brand_tier", meta.BrandTier)
	fi.DistrictEnc = ms.Encode("hotel_district", meta.HotelDistrict)
	fi.FuncTierEnc = ms.Encode("district_functional_tier", meta.DistrictFunctional)
	fi.IsChain = meta.IsChain

	// 时间特征
	dow := int(date.Weekday()+6) % 7
	fi.DayOfWeek = dow
	fi.Month = int(date.Month())
	fi.DayOfMonth = date.Day()
	fi.Quarter = (int(date.Month())-1)/3 + 1
	if dow >= 5 {
		fi.IsWeekend = 1
	}
	fi.MonthSin = math.Sin(2 * math.Pi * float64(fi.Month) / 12)
	fi.MonthCos = math.Cos(2 * math.Pi * float64(fi.Month) / 12)
	fi.DowSin = math.Sin(2 * math.Pi * float64(dow) / 7)
	fi.DowCos = math.Cos(2 * math.Pi * float64(dow) / 7)

	// 天气使用默认值
	fi.Tavg = 13.0
	fi.Tmin = 7.0
	fi.Tmax = 20.0
	fi.Prcp = 0.0
	fi.Snow = 0.0
	fi.Wdir = 180.0
	fi.Wspd = 10.0
	fi.Wpgt = 15.0
	fi.Pres = 1013.0
	fi.Tsun = 200.0

	// Lag / Rolling
	sorted := make([]HistoryRecord, len(history))
	copy(sorted, history)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].RecordDate < sorted[j].RecordDate
	})

	var past []float64
	for _, r := range sorted {
		d, err := time.Parse("2006-01-02", r.RecordDate)
		if err != nil {
			continue
		}
		if !d.Before(date) {
			break
		}
		past = append(past, r.OccupancyRate)
	}

	lagDays := ms.XGBFeatMeta.LagDays
	rollWins := ms.XGBFeatMeta.RollWindows
	fallback := 50.0
	if len(past) > 0 {
		fallback = past[len(past)-1]
	}

	fi.Lags = make(map[int]float64)
	fi.RollingMean = make(map[int]float64)
	fi.RollingStd = make(map[int]float64)
	fi.RollingMax = make(map[int]float64)
	fi.RollingMin = make(map[int]float64)

	for _, lag := range lagDays {
		idx := len(past) - lag
		if idx >= 0 {
			fi.Lags[lag] = past[idx]
		} else {
			fi.Lags[lag] = fallback
		}
	}

	for _, w := range rollWins {
		start := len(past) - w
		if start < 0 {
			start = 0
		}
		window := past[start:]
		if len(window) == 0 {
			fi.RollingMean[w] = fallback
			fi.RollingStd[w] = 0
			fi.RollingMax[w] = fallback
			fi.RollingMin[w] = fallback
			continue
		}
		sum, mn, mx := 0.0, window[0], window[0]
		for _, v := range window {
			sum += v
			if v < mn {
				mn = v
			}
			if v > mx {
				mx = v
			}
		}
		mean := sum / float64(len(window))
		fi.RollingMean[w] = mean
		fi.RollingMax[w] = mx
		fi.RollingMin[w] = mn

		// std
		if len(window) < 2 {
			fi.RollingStd[w] = 0
		} else {
			sumSq := 0.0
			for _, v := range window {
				sumSq += (v - mean) * (v - mean)
			}
			fi.RollingStd[w] = math.Sqrt(sumSq / float64(len(window)-1))
		}
	}

	return fi
}

// ToVector 按 feature_order 组装成 float64 切片
func (fi *FeatureInput) ToVector(featureOrder []string) []float64 {
	vec := make([]float64, len(featureOrder))
	for i, name := range featureOrder {
		vec[i] = fi.get(name)
	}
	return vec
}

func (fi *FeatureInput) get(name string) float64 {
	switch name {
	case "hotel_id_enc":
		return float64(fi.HotelIDEnc)
	case "brand_tier_enc":
		return float64(fi.BrandEnc)
	case "hotel_district_enc":
		return float64(fi.DistrictEnc)
	case "district_functional_tier_enc":
		return float64(fi.FuncTierEnc)
	case "is_chain":
		return float64(fi.IsChain)
	case "day_of_week":
		return float64(fi.DayOfWeek)
	case "month":
		return float64(fi.Month)
	case "day_of_month":
		return float64(fi.DayOfMonth)
	case "quarter":
		return float64(fi.Quarter)
	case "is_weekend":
		return float64(fi.IsWeekend)
	case "month_sin":
		return fi.MonthSin
	case "month_cos":
		return fi.MonthCos
	case "dow_sin":
		return fi.DowSin
	case "dow_cos":
		return fi.DowCos
	case "is_public_holiday":
		return float64(fi.IsPublicHoliday)
	case "is_workday":
		return float64(fi.IsWorkday)
	case "is_school_vacation":
		return float64(fi.IsSchoolVacation)
	case "tavg":
		return fi.Tavg
	case "tmin":
		return fi.Tmin
	case "tmax":
		return fi.Tmax
	case "prcp":
		return fi.Prcp
	case "snow":
		return fi.Snow
	case "wdir":
		return fi.Wdir
	case "wspd":
		return fi.Wspd
	case "wpgt":
		return fi.Wpgt
	case "pres":
		return fi.Pres
	case "tsun":
		return fi.Tsun
	default:
		// lag_N
		for _, lag := range []int{1, 2, 3, 7, 14, 21} {
			if name == fmt.Sprintf("lag_%d", lag) {
				return fi.Lags[lag]
			}
		}
		for _, w := range []int{7, 14, 30} {
			if name == fmt.Sprintf("rolling_mean_%d", w) {
				return fi.RollingMean[w]
			}
			if name == fmt.Sprintf("rolling_std_%d", w) {
				return fi.RollingStd[w]
			}
			if name == fmt.Sprintf("rolling_max_%d", w) {
				return fi.RollingMax[w]
			}
			if name == fmt.Sprintf("rolling_min_%d", w) {
				return fi.RollingMin[w]
			}
		}
		return 0
	}
}
