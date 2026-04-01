# 酒店入住率预测系统

[English](README.md) | 简体中文

---

## 概览

一个实用的酒店入住率时间序列预测系统。

* 模型：ARIMA, XGBoost
* 后端：Go（服务）+ Python（训练）
* 特征：时间、日历、天气、滞后、滚动
* 包含 API + 简单前端可视化

---

## 核心功能

* 时间序列 → 监督学习（滞后 & 滚动特征）
* 全局 XGBoost 模型（跨酒店学习）
* 特征消融实验
* SHAP 特征解释
* Go API 服务

---

## 数据

* 47 家酒店
* 约 66k 日记录
* 数据范围：2021-12 ~ 2025-11
* 预测目标：次日入住率

---

## 技术栈

* Python：pandas, xgboost, statsmodels, shap
* Go：REST API 推理服务
* 前端：HTML + Chart.js

---

## 快速开始

### 1. 模型训练

```bash
cd python
python train_all.py
````

### 2. 启动 API

```bash
go run cmd/server/main.go -python <venv>/python.exe
```

### 3. 打开前端界面

打开 `http://localhost:8080`

---

## API 接口

* `/predict` – 预测结果
* `/history` – 历史数据
* `/metrics` – 模型性能指标
* `/health` – 服务状态


---

## 后续计划

* 实时天气 API 接入
* 多步预测
* LightGBM / CatBoost 对比实验

---

## 截图示例

### 预测结果

![预测结果](docs/screenshot/predict.png)

### 历史数据

![历史数据](docs/screenshot/history.png)

### 模型指标

![模型指标](docs/screenshot/metrics.png)

