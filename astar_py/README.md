# Description

This is a fast cpp implementation of A* with python bindings.
The package includes some grid utilities as well.

## Usage

* Compile

Replace the `/miniconda/envs/rg_sddm/bin/python` with your own python3 in conda environments
Example

  ```sh
  mkdir build && cd build
  cmake -DPYTHON_EXECUTABLE=/miniconda/envs/rg_sddm/bin/python ..
  && make && cd ..
  ```

* Copy library to `[repo path]/src/mylib`

Example

  ```sh
  mv ../lib/*.so /test/icra_sim_clean/src/mylib
  ```