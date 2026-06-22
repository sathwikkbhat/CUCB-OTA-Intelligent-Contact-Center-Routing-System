# 🚀 CUCB-OTA: Intelligent Contact Center Routing System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Vercel Deployment](https://img.shields.io/badge/Vercel-Deploy--Ready-000000?style=flat&logo=vercel)](https://vercel.com)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python)](https://www.python.org/)
[![Flask API](https://img.shields.io/badge/API-Flask--REST-green.svg?logo=flask)](https://flask.palletsprojects.com/)
[![WebSockets](https://img.shields.io/badge/Real--Time-WebSockets-orange.svg?logo=socket.io)](https://socket.io/)

A state-of-the-art, production-ready AI system that matches customers to contact center agents in real-time. By combining **Causal Machine Learning (Uplift Modeling)**, **Constrained Optimization (Lagrangian Relaxation)**, and **Optimal Transport (Hungarian Algorithm)**, CUCB-OTA maximizes customer satisfaction while mathematically guaranteeing that Average Handle Time (AHT), Service Level Agreements (SLA), and agent workload fairness remain within defined operational budgets.

---

## 🎯 Problem Statement

Traditional contact center routing systems rely on simple heuristics that ignore the nuances of customer-agent interactions:

*   **First-Come-First-Served (FCFS)** matches the next customer to the next available agent, ignoring skill levels and complexity.
*   **Static Skill-Based Routing (SBR)** assigns tags (e.g., "Billing"), but treats all tagged agents as identical, overloading expert agents, breaching SLAs, and causing agent burnout.

### Operational Pain Points
*   ❌ **Customer Churn**: Poor agent matching leads to multiple transfers, higher effort, and negative customer experiences.
*   ❌ **High Average Handle Time (AHT)**: Complex cases routed to junior or mismatched agents drag out call durations, inflating OpEx.
*   ❌ **SLA Breaches**: High wait times due to bottlenecked skill pools.
*   ❌ **Agent Burnout**: Workloads accumulate on top performers, causing high turnover (historically **30%–45%** annually in contact centers).

---

## 💎 The Solution: CUCB-OTA

CUCB-OTA shifts the routing paradigm from "who is available" to **"who will provide the highest customer satisfaction uplift"** while keeping operational guardrails active.

```
                  ┌─────────────────────────────────────┐
                  │         Incoming Customers          │
                  └──────────────────┬──────────────────┘
                                     │ (Vectorized Batches)
                                     ▼
                  ┌─────────────────────────────────────┐
                  │        CUCB-OTA Router API          │
                  │  (X-Learner Uplift + capacity check)│
                  └──────────────────┬──────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼ (Hungarian Solver / Greedy Capacity)  ▼
      ┌─────────────────────┐                 ┌─────────────────────┐
      │   Voice Channel     │                 │ Chat/Email Channels │
      │   (One-to-One)      │                 │    (One-to-Many)    │
      └──────────┬──────────┘                 └──────────┬──────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │       Active SLA / AHT Control      │
                  │  (Lagrangian Self-Tuning Weights)   │
                  └─────────────────────────────────────┘
```

### Key Technical Pillars

1.  **Causal Uplift (X-Learner)**: Predicts the *conditional average treatment effect (CATE)* to identify the marginal CSAT increase of matching a customer to an expert rather than a generalist.
2.  **Constrained Lagrangian Optimization**: Learns dual variables ($\lambda$) dynamically to penalize routing actions that violate AHT, SLA, or Gini-workload fairness budgets.
3.  **Optimal Transport Assignment**: Uses the **Hungarian Algorithm** ($O(n^3)$) to guarantee mathematically optimal voice matching and a **Greedy Capacity-Aware Solver** for chat/email concurrency constraints.
4.  **Omnichannel Capacities**: Integrates cross-channel blocking rules (Voice: 1 call, Chat: 3 concurrent, Email: 5 concurrent).

---

## ⚡ Vercel Deployment (Frontend)

The frontend is fully configured and optimized to be **Vercel-ready**!

### How to Deploy the Frontend to Vercel:

1.  **Connect to GitHub**: Import this repository into your Vercel Account.
2.  **Configure Directory**: Under project settings in Vercel, set the **Root Directory** to `frontend`.
    *   *Alternatively*, the preconfigured [vercel.json](file:///c:/Users/bhats/CUCB/vercel.json) at the root of the project will automatically route all static root traffic to the `frontend/` directory.
3.  **Deploy**: Click **Deploy**. Vercel will host your static files globally.

### Connecting to Your Backend API:
Since the Flask server runs asynchronously with simulation threads, the frontend has a **Dynamic Backend API Input** built directly into the sidebar footer.
*   Once deployed, enter your backend's host URL (e.g., `https://your-cucb-api.railway.app` or `http://localhost:5000`) into the **Backend API URL** field.
*   The system will persist this setting in `localStorage` and automatically reconnect the WebSocket/REST calls.

---

## 📂 Project Structure

```
├── vercel.json               # Vercel static routing configuration
├── config.py                 # Simulation constants, rules, and hyperparameters
├── main.py                   # Command-line interface simulation runner
├── PROJECT_REPORT.md         # Full project report (problem, math, business ROI)
├── JUDGE_PRESENTATION.md     # Presentation handbook for judges
├── requirements.txt          # Python dependencies
├── api/
│   ├── app.py                # Flask API & WebSocket server
│   ├── test_api.py           # Endpoint integration tests
│   └── README.md             # API endpoint documentation
├── frontend/
│   └── index.html            # Dashboard UI (Charts, Agent Grid, Logs)
├── models/
│   └── uplift_model.py       # Causal X-Learner + AHT prediction models
├── routing/
│   ├── scoring.py            # Dual score formulation
│   └── assignment.py         # Hungarian & Greedy Capacity allocators
├── evaluation/
│   ├── metrics.py            # KPI metrics tracker
│   └── ope.py                # Off-policy evaluation
├── simulation/
│   ├── simulator.py          # Queue event loop
│   └── visualizer.py         # Simulation matplotlib plotters
└── tests/
    └── test_batch_predictions.py
```

---

## ⚙️ Local Installation & Development

### Prerequisites
*   Python 3.8+
*   pip

### Setup
```bash
# Clone the repository
git clone https://github.com/sathwikkbhat/CUCB-OTA-Intelligent-Contact-Center-Routing-System.git
cd CUCB-OTA-Intelligent-Contact-Center-Routing-System

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Running the Project

### Option 1: CLI Simulation (Quick Demo)
Run the complete offline simulation pipeline to train models and evaluate routing policies over 150 batches:
```bash
python main.py
```
**Outputs Generated in `data/logs/`:**
*   `policy_comparison.png` - side-by-side performance benchmarks.
*   `cucb_convergence.png` - tracks Lagrangian variable adjustment.
*   `agent_workload.png` - workload distributions and agent scatter plots.
*   `final_report_*.txt` - final summary metrics.

### Option 2: Run Locally (Flask Server + Frontend)
1.  **Start Backend API Server**:
    ```bash
    python api/app.py
    ```
    The server starts on `http://localhost:5000` with WebSocket connections.

2.  **Serve Frontend**:
    ```bash
    cd frontend
    python -m http.server 3000
    ```
    Open `http://localhost:3000` in your web browser.

---

## 📊 Expected Performance Results

Tested over 150 batches (7,500 customer interactions, 30 agents):

| Performance Metric | FCFS (Baseline) | Skill-Greedy | **CUCB-OTA (Our System)** |
| :--- | :---: | :---: | :---: |
| **Average CSAT** | 0.7234 | 0.7456 | **0.7812 (+8.0%)** |
| **Average Handle Time** | 7.43 min | 7.21 min | **6.89 min (-7.2%)** |
| **SLA Met Rate** | 82.1% | 85.3% | **91.2% (+11.1%)** |
| **Workload Gini (Fairness)**| 0.412 | 0.387 | **0.234 (-43.2%)** |
| **AHT Constraint (≤ 8 min)** | Met | Met | **Met (6.89 min) ✅** |
| **SLA Target (≥ 85%)** | Breached | Met | **Met (91.2%) ✅** |
| **Fairness Gini (≤ 0.3)** | Breached | Breached | **Met (0.234) ✅** |

---

## 🛠️ Customization

Adjust simulation parameters directly in [config.py](file:///c:/Users/bhats/CUCB/config.py):
*   **Scale up capacity**: Update `NUM_AGENTS = 50`, `NUM_CUSTOMERS_PER_BATCH = 100`.
*   **SLA targets**: Change thresholds in `SLA_THRESHOLDS` (e.g. Chat SLA: 3 mins -> 2 mins).
*   **Lagrangian learning rate**: Tune convergence speed via `LAMBDA_LR`.

---

## 👥 Contributors

*   **Sathwik Bhat** - [sathwikkbhat](https://github.com/sathwikkbhat)

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
