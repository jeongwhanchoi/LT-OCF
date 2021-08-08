# LT-OCF: Learnable-Time ODE-based Collaborative Filtering

## Our proposed LT-OCF

 <img src="img/lt-ocf.png" height="250">

### Our proposed dual co-evolving ODE

<img src="img/dualres.png" height="250">

---

## Setup Python environment for LT-OCF

### Install python environment

```bash
conda env create -f environment.yml   
```

### Activate environment
```bash
conda activate lt-ocf
```

---

## Reproducibility
### Usage

#### In terminal
- Run the shell file (at the root of the project)
```bash
# run lt-ocf (gowalla dataset, rk4 solver, learnable time)
sh ltocf_gowalla_rk4.sh
```
```bash
# run lt-ocf (gowalla dataset, rk4 solver, fixed time)
sh ltocf_gowalla_rk4_fixed.sh
```

#### Arguments (see more arguments in `parse.py`)
- gpuid
    - default: 0
- dataset
    - gowalla, yelp2018, amazon-book
- model
    - **ltocf**
- solver
    - euler, **rk4**, implicit_adams, dopri5
- adjoint
    - **False**, True
- K
    - 1, 2, 3, **4**
- learnable_time
    - True, False
- dual_res
    - **False**, True