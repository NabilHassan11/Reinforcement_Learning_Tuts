# 🤖 Reinforcement Learning Tutorials:🧠

# 1️⃣ MDP & Dynamic Programming Tutorial

Welcome to the **MDP_DP** folder! This section of the repository focuses on **Markov Decision Processes (MDPs)** **Bellman Optimality Equations** and **Dynamic Programming (DP)** techniques in Reinforcement Learning.

## 🏭 Warehouse Maintenance Optimization with MDPs & Dynamic Programming ⚙️

![MDP Diagram](![image](https://github.com/user-attachments/assets/04281750-c46e-4c50-93a4-f2284c632e57)
)
) *Visualization of the warehouse maintenance MDP*

### 🔍 Problem Statement
**Task:** Optimize maintenance strategies for robotic systems in a warehouse using Reinforcement Learning.  
**Challenge:** Prevent robotic system failures while minimizing downtime through optimal maintenance scheduling.  
**Goal:** Find the policy that maximizes long-term rewards using **Markov Decision Processes (MDPs)** and **Dynamic Programming**.

---

### 📦 Environment Specification (MDP)

#### 🧩 States
| State | Symbol | Description | 
|-------|--------|-------------|
| Operational | 🟢 **O** | Robot functioning normally |
| Needs Maintenance | 🟡 **NM** | Early wear signs but operational |
| Maintenance | 🔵 **M** | Undergoing scheduled maintenance | 
| Broken | 🔴 **B** | Critical failure state |

#### 🎮 Actions
| Action | Symbol | Description |
|--------|--------|-------------|
| Perform Maintenance | PM | Proactive maintenance |
| Continue Operation | CO | No intervention |
| Schedule Maintenance | SM | Plan future maintenance |
| Repair | R | Fix broken robot |

---

## ⚡ Key Features
```python
# Highlighted code feature: Stochastic transitions
def get_transitions(state, action):
    if state == "NM" and action == "CO":
        # 50% chance breakdown, 50% stay degraded
        return [('B', 0.5), ('NM', 0.5)]
```

🌐 Environment Dynamics
|State	|Action	|Transitions	|Reward|
|-------|-------|---------------|------|
|🟢 O	|CO	|80% stay O, 20% → 🟡 NM	|+20 if stays O|
|🟡 NM	|PM	|100% → 🔵 M	|0|
|🔵 M	|SM	|100% → 🟢 O	|+20|
|🔴 B	|R	|100% → 🟢 O	|+20|

⛔Note: The Environment Dynamics is not fully written here you can refer for it from the code 

---

## 🗂️ Folder Contents
Explore algorithms and implementations for solving MDPs using DP methods:
- **Policy Iteration** 🔄
- **Value Iteration** ⚡
- **Policy Evaluation** 📊

### 📜 Code Files:
| File | Description |
|------|-------------|
| [`policy_iteration.py`](./MDP_DP/policy_iteration.py) | Policy Iteration algorithm implementation |
| [`value_iteration.py`](./MDP_DP/value_iteration.py) | Value Iteration algorithm implementation |
| [`Lecture_3_PolicyEval.py`](./MDP_DP/Lecture_3_PolicyEval.py) | Policy Evaluation demonstration | 

---

## 🚀 Quick Start
### Prerequisites
- Python 3.8+
- NumPy (`pip install numpy`)

### Installation
```bash
git clone https://github.com/NabilHassan11/Reinforcement_Learning_Tuts.git
cd Reinforcement_Learning_Tuts/MDP_DP
