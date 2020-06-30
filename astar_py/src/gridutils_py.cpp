#include <vector>
#include <cmath>
#include <iterator>     // std::distance
#include <limits>
#include <algorithm>    // std::min_element, std::max_element



inline int meters2cells(double datam, double min, double res) { return static_cast<int>(std::floor((datam - min) / res)); }
inline double cells2meters(int datac, double min, double res) { return (static_cast<double>(datac) + 0.5) * res + min; }
inline std::vector<int> meters2cells(std::vector<double> const &datam,
                                     std::vector<double> const &dim_min,
                                     std::vector<double> const &res) {
  std::vector<int> datac(datam.size());
  for (unsigned k = 0; k < datam.size(); ++k)
    datac[k] = meters2cells(datam[k], dim_min[k], res[k]);
  return datac;
}
inline std::vector<double> cells2meters(std::vector<int> const &datac,
                                        std::vector<double> const &dim_min,
                                        std::vector<double> const &res) {
  std::vector<double> datam(datac.size());
  for (unsigned k = 0; k < datac.size(); ++k)
    datam[k] = cells2meters(datac[k], dim_min[k], res[k]);
  return datam;
}

// Row major order as in C++
inline std::size_t subv2ind_rowmajor(std::vector<int>::const_iterator const &datac_begin,
                                     std::vector<int>::const_iterator const &datac_end,
                                     std::vector<int>::const_iterator const &size_begin,
                                     std::vector<int>::const_iterator const &size_end) {
  if (datac_end <= datac_begin || size_end <= size_begin) return -1;
  std::size_t idx = *(datac_end - 1);
  std::size_t prod = 1;
  std::vector<int>::const_iterator it1 = datac_end - 2;
  std::vector<int>::const_iterator it2 = size_end - 1;
  for (; it1 != (datac_begin - 1) && it2 != size_begin; --it1, --it2) {
    prod *= (*it2);
    idx += prod * (*it1);
  }
  return idx;
}

inline std::vector<int> ind2subv_rowmajor(std::size_t ind,
                                          const std::vector<int>::const_iterator &size_begin,
                                          const std::vector<int>::const_iterator &size_end) {
  const std::size_t ndims = std::distance(size_begin, size_end);
  std::vector<int> subv(ndims);
  std::vector<int>::const_iterator it = size_end - 1;
  for (int k = ndims - 1; k >= 0; --k, --it) {
    subv[k] = ind % (*it);
    ind -= subv[k];
    ind /= (*it);
  }
  return subv;
}

// Column major order as in MATLAB
inline std::size_t subv2ind_colmajor(std::vector<int>::const_iterator const &datac_begin,
                                     std::vector<int>::const_iterator const &datac_end,
                                     std::vector<int>::const_iterator const &size_begin,
                                     std::vector<int>::const_iterator const &size_end) {
  if (datac_end <= datac_begin || size_end <= size_begin) return -1;
  std::size_t idx = *datac_begin;
  std::size_t prod = 1;
  std::vector<int>::const_iterator it1 = datac_begin + 1;
  std::vector<int>::const_iterator it2 = size_begin;
  for (; it1 != datac_end && it2 != (size_end - 1); ++it1, ++it2) {
    prod *= (*it2);
    idx += (*it1) * prod;
  }
  return idx;
}

inline std::vector<int> ind2subv_colmajor(std::size_t ind,
                                          const std::vector<int>::const_iterator &size_begin,
                                          const std::vector<int>::const_iterator &size_end) {
  const std::size_t ndims = std::distance(size_begin, size_end);
  std::vector<int> subv(ndims);
  std::vector<int>::const_iterator it = size_begin;
  for (std::size_t k = 0; k < ndims; ++k, ++it) {
    subv[k] = ind % (*it);
    ind -= subv[k];
    ind /= (*it);
  }
  return subv;
}


inline void bresenhamStep(double dv, double sv, int svc, double vmin, double vres,
                          int &stepV, double &tDeltaV, double &tMaxV)
{
  if (dv > 0) {
    stepV = 1;
    tDeltaV = vres / dv;
    tMaxV = (vmin + (svc + 1) * vres - sv) / dv; // parametric distance until the first crossing
  } else if (dv < 0) {
    stepV = -1;
    tDeltaV = vres / -dv;
    tMaxV = (vmin + svc * vres - sv) / dv;
  } else {
    stepV = 0;
    tDeltaV = 0.0;
    tMaxV = std::numeric_limits<double>::infinity(); // the line doesn't cross the next plane
  }
}

inline std::vector<std::vector<int>> 
bresenham( const std::vector<double>& start,
           const std::vector<double>& end,
           const std::vector<double>& gridmin,
           const std::vector<double>& gridres )
{  
  // find start and end cells
  std::vector<int> start_cell = meters2cells(start,gridmin,gridres);
  std::vector<int> end_cell =  meters2cells(end,gridmin,gridres);
  std::vector<double> diff(start.size());
  std::transform(end.begin(), end.end(), start.begin(), diff.begin(), std::minus<double>());
  std::vector<int> step(start_cell.size()); // direction of grid traversal
  std::vector<double> tDelta(start_cell.size()); // parametric step size along different dimensions
  std::vector<double> tMax(start_cell.size()); // used to determine the dim of the next step along the line
  for( size_t k = 0; k < diff.size(); ++k )
    bresenhamStep(diff[k], start[k], start_cell[k], gridmin[k], gridres[k], step[k], tDelta[k], tMax[k]);

  // Add initial voxel to the list
  std::vector<std::vector<int>> cellidx;
  cellidx.push_back(start_cell); 
  while( start_cell != end_cell )
  {
    auto min_idx = std::distance(tMax.begin(), std::min_element(tMax.begin(), tMax.end()));
    start_cell[min_idx] += step[min_idx];
    tMax[min_idx] += tDelta[min_idx];
    cellidx.push_back(start_cell);
  }
  return cellidx;
}

template<class T>
inline std::vector<T>
inflateMap( const std::vector<T>& map_cost,
            const std::vector<int>& map_size,
            const std::vector<double>& map_resolution,
            bool is_rowmajor,
            const std::vector<double>& inflation_radius )
{
  
  // find the number of cells to inflate per dimension
  std::vector<int> extents(map_resolution.size());
  size_t num_ns = 1;
  for( size_t i = 0; i < map_resolution.size(); ++i )
  {
    extents[i] = 2*static_cast<int>(std::ceil(inflation_radius[i]/map_resolution[i]))+1;
    num_ns *= extents[i];
  }
  
  // compute the neighbor coordinates
  std::vector<std::vector<int>> ns;
  ns.reserve(num_ns);
  for( size_t i = 0; i < num_ns; ++i )
  {
    // convert the linear index of every potential neighbor
    // to an index in [-(extents[i]-1)/2,(extents[i]-1)/2]
    // and check if this falls within the axis aligned ellipsoid
    // defined by inflation_radius
    std::vector<int> subv = ind2subv_colmajor(i,extents.begin(),extents.end());
    //std::cout << subv << std::endl;
    double d = 0.0; // ellipsoid radius
    for( size_t k = 0; k < map_size.size(); ++k )
    {
      subv[k] -= (extents[k]-1)/2; // add offset (i.e., cells2meters)
      double tmp = subv[k]*(map_resolution[k] / inflation_radius[k]);
      d += tmp*tmp;
    }
    
    if( 0.0 < d && d <= 1.000001 )
      ns.push_back(subv);
  }

  std::vector<T> inflated_cost(map_cost);
  for( size_t i = 0; i < map_cost.size(); ++i )
    if( map_cost[i] > T(0) )
    {
      // get coordinates of ith entry
      std::vector<int> subv = is_rowmajor ? ind2subv_rowmajor(i,map_size.begin(),map_size.end()):    
                                            ind2subv_colmajor(i,map_size.begin(),map_size.end());
      // find the neighbors and make them occupied
      for (const auto &it : ns)
      {
        bool valid = true;
        std::vector<int> neib_coord(it);
        for( size_t k = 0; k < map_size.size(); ++k )
        {
          neib_coord[k] += subv[k];
          if( neib_coord[k] < 0 || neib_coord[k] >= map_size[k] )
          {
            valid = false;
            break;
          }
        }
        if( valid )
          inflated_cost[is_rowmajor ? subv2ind_rowmajor(neib_coord.begin(),neib_coord.end(),map_size.begin(),map_size.end()):    
                        subv2ind_colmajor(neib_coord.begin(),neib_coord.end(),map_size.begin(),map_size.end())] = map_cost[i];
      }
    }
  return inflated_cost;
}



#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
PYBIND11_MODULE(gridutils_py, m)
{
  m.doc() = "Pybind11 plugin for gridmap utilities";

  m.def("bresenham", &bresenham,
    "Bresenham's line algorithm.\n\n"
    "Input:\n"
    "\tstart: nd vector containing the start coordinates\n"
    "\tend: nd vector containing the end coordinates\n"
    "\tgridmin: nd vector containing the lower left corner of the grid (in meters)\n"
    "\tgridres: nd vector containing the grid resolution (in meters)\n",
    pybind11::arg("start"),
    pybind11::arg("end"),
    pybind11::arg("gridmin"),
    pybind11::arg("gridres"));

  m.def("inflateMap", &inflateMap<int>,
    "Inflate an obstacle map with a specified radius.\n\n"
    "Input:\n"
    "\tcMap: nd vector containing the costmap\n"
    "\tmap_size: nd vector containing the costmap dimensions\n"
    "\tmap_resolution: nd vector containing the grid resolution (in meters)\n"
    "\tis_rowmajor: bool specifying if the map is row or column major organized\n",
    "\tinflation_radius: nd vector specifying the inflation in each axis\n",
    pybind11::arg("cMap"),
    pybind11::arg("map_size"),
    pybind11::arg("map_resolution"),
    pybind11::arg("is_rowmajor"),
    pybind11::arg("inflation_radius"));

  m.def("inflateMap", &inflateMap<double>,
    "Inflate an obstacle map with a specified radius.\n\n"
    "Input:\n"
    "\tcMap: nd vector containing the costmap\n"
    "\tmap_size: nd vector containing the costmap dimensions\n"
    "\tmap_resolution: nd vector containing the grid resolution (in meters)\n"
    "\tis_rowmajor: bool specifying if the map is row or column major organized\n",
    "\tinflation_radius: nd vector specifying the inflation in each axis\n",
    pybind11::arg("cMap"),
    pybind11::arg("map_size"),
    pybind11::arg("map_resolution"),
    pybind11::arg("is_rowmajor"),
    pybind11::arg("inflation_radius"));
    
  m.attr("__version__") = "dev";
}

