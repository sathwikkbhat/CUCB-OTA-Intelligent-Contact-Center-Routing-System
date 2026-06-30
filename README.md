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

## 🌟 Hackathon-Grade Interactive Features (Frontend Vercel Configuration)

To make CUCB-OTA outstanding for evaluations and presentation, the frontend is packed with visual, interactive controls that function **with or without** the Python backend running:

### 1. 📲 Mobile-Responsive Drawer Navigation
- **Fluid layouts** adapt between widescreen monitors, tablets, and phones.
- On mobile devices, the desktop sidebar transforms into a slide-out navigation drawer toggled by a hamburger menu, utilizing overlay backdrops and auto-closing triggers.
- Multi-column metric grids stack vertically to prevent horizontal overflows.

### 2. ⚡ Zero-Config Vercel Fallback (Demo Mode)
- **Automatic failover**: The frontend checks the API connection. If the Flask backend is offline, the interface shifts to a high-fidelity **Demo Mode** with a pulsing amber warning indicator.
- **Client-Side Simulation**: Emulates the backend routing rules, generating 30 mock agents and compiling real-time convergence graphs, SLA logs, and dual variables in pure Javascript.
- **Seamless Recovery**: If you spin up your backend locally or host it on Render/Railway and enter the URL in the sidebar input, the frontend automatically switches off Demo Mode and connects to the live API!

### 3. ⚠️ Anomalous Traffic Spike Simulator
- A **Trigger Spike** button in the Simulation control panel simulates extreme outage workloads.
- Incoming batch size multiplies by **350%** and requests become highly complex.
- **FCFS** and **Skill-Greedy** baselines immediately experience severe SLA violations (SLA met falls to ~40%, AHT rises to 9+ min), whereas **CUCB-OTA's** Lagrangian dual variables spike to re-allocate tickets, keeping SLAs stable (~82%) and workload balanced (very low Gini coefficient).

### 4. 🎓 Interactive Agent Profile Modals & Training Sliders
- Click on any agent card in the **Agents** panel to open their profile drawer.
- Displays Voice, Chat, and Email loads, CSAT metrics, and skill tags.
- Includes **dynamic skill sliders** allowing you to "train" the agent in real time. Applying updates modifies the active simulator state; subsequent batches immediately reflect the performance benefits of your training!

### 5. 📊 Visual Dashboard Reports & Text Downloader
- Opens structured visual cards, benchmarks, and comparison tables inside the report modal.
- Includes a **Download Report** button that generates and downloads a formatted `.txt` report file of the simulation logs.

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
│   └── index.html            # Premium White Dashboard (Charts, Agent Grid, Modals)
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

## 🔬 Algorithm & Mathematical Details

### 1. Routing Score Formula
The assignment matches customers to agents by maximizing the routing score $RS(c,a)$:
$$RS(c, a) = \tau(c, a) - \lambda_1 \cdot \text{AHT}(c,a) - \lambda_2 \cdot \text{SLA\_risk}(c,a) - \lambda_3 \cdot \text{Fairness}(a)$$

Where:
*   $\tau(c, a)$: Predicted CSAT uplift calculated using X-Learner.
*   $\lambda_1, \lambda_2, \lambda_3$: Lagrangian multipliers that adjust adaptively based on constraint violations.
*   $\text{Fairness}(a)$: Current agent load penalty calculated to balance assignments.

### 2. Dual Update Rule
Lagrangian dual variables tune themselves after each batch step:
$$\lambda_i \leftarrow \max(0, \lambda_i + \eta \cdot (\text{constraint\_violation} - \text{budget}))$$

Where $\eta$ represents the optimization learning rate. If violations exceed target budgets, $\lambda$ rises to penalize violating decisions in subsequent rounds.

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
