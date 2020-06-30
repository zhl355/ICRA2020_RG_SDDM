# Reference Governor Safe Control with SDDM

This repository contains code for ICRA 2020 `Fast and Safe Path-Following Control using a State-Dependent Directional Metric`.  Reference governor framework equipped with novel metric enables fast and safe navigation in unknown complex environments. This code is primarily developed by Zhichao Li, a PhD student in the [Exsitential Robototics Laboratory](http://erl.ucsd.edu/).

## Dependency

The code is test on `Ubuntu 16.04/18.04 LTS` on a desktop computer (Intel i7-8700K CPU  + 32 GB RAM) `Anaconda (Python 3.6.7) + Spyder IDE` is recommneded. IPython console in Spyder can visualize most result in a pleasant way. I will list the most important packages, you can check all packages in `rg_sddm.yml`.

* third-party packages
  + [vtkplotter 2019.4.4](https://vtkplotter.embl.es/)
  + [mosek solver 9.0](https://www.mosek.com/)
  + [cvxpy 1.0.24](https://www.cvxpy.org/)
  + [trimesh 3.1.5](https://github.com/mikedh/trimesh)

* Other dependencies
  + A* planner compile and copy `*.so` to `/src/mylib` see README in `astar_py`
  + Eigen3  `sudo apt install -y libeigen3-dev`
  + Boost `sudo apt install-y libboost-dev`

To replicate the testing environment, we recommend using `Anaconda` to create
a virtual environment as follows. For both approaches, you need to obtain a
**MOSEK solver license** (Academic free license) and install it properly according to
replied email.

### Use Docker

docker build -t rg_sddm -f Dockerfile-RG_SDDM ./
docker run -it --rm rg_sddm

### Mannual setup

  ```sh
  conda create --name rg_sddm python=3.6
  conda activate rg_sddm
  conda install -c conda-forge scikit-image shapely rtree pyembree
  conda install -c mosek mosek
  conda install scipy matplotlib nose
  conda install -c anaconda spyder
  pip install cvxpy trimesh[all] vtkplotter==2019.4.4
  ```

## Usage

To replicate the main result, please use the following instruction. I avoid using arg parser and make the code very easy to achieve different functionalities by just toggling comments of certain lines. For simplicity,
the usage is omitted, and more detail can be found in the main functions.
Run all codes within `src` folder. All log files are save in `log` folder and all figures are saved in `sim_figs` folder.

* Evaluate sparse known circular environment
  + SDDM with **c1=1, c2=4** (default), [about 2.5 minutes]

    **Run** `main_sparse_known.py` in Spyder, or

    ```py
    python main_sparse_known.py
    ```

  + Baseline Euclidean norm **c1=1, c2=1** (**change line 45**) , [about 15 secs].

    ```py
    python main_sparse_known.py
    ```

  + To view more result, please uncomment the correpsonding lines (around line 160)

    ```py
    # %% Show the result
    viewer = GovLogViewer(logname_full)
    # viewer.play_animation(0.01, save_fig=True, fig_sample_rate=1)
    viewer.show_trajectory()
    # viewer.show_robot_gov_stat()
    # viewer.compare_eta_max()
    ```

* Evaluate dense cluttered unknown environment [about 2.5 minutes]
    **Run** `main_dense_unknown.py` in Spyder, or

    ```py
    python main_dense_unknown.py
    ```

* Create and Display 3D video for dense environment simulation.[about 1.5 minutes]

  ```py
  python main_dense_video3D.py
  ```

* Generating 2D top view video frames [about 21 minutes]
  + Generating PNG frames [about 20 minutes without GPU]

    ```py
    python main_dense_video2D.py.
    ```

  + Use `ffmpeg` make video [1 minutes]

    ```sh
    cd [this repo]/sim_figs/dense_env2D
    ffmpeg -i LMOG_1%04d.png -vcodec libx264 -b 10000k dense_env2D.avi
    ```

## Citing this work

If you find the ideas and code implementation are useful for your own work, please check out the following papers and cite the appropriate ones:

* Reference Governor + SDDM on simplified dynamics [ICRA 2020 Prepint](https://arxiv.org/abs/2002.02038)
* Referene Governor on stochastic nonlinear system [New extension on ArXiv](https://arxiv.org/pdf/2005.06694.pdf)
