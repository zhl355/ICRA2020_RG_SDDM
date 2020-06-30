
#include <vector>
#include <list>
#include <unordered_map>                  // hash
#include <boost/heap/d_ary_heap.hpp>      // heap

// pybind11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
//#include <pybind11/eigen.h>

namespace astar
{
  static constexpr double infCost = std::numeric_limits<double>::infinity();
  static constexpr double sqrtTwoCost = std::sqrt(2);
  static constexpr double sqrtThreeCost = std::sqrt(3);
  
  template <class T>
  struct compareAStates
  {
    bool operator()(T* a1, T* a2) const
    {
      double f1 = a1->g + a1->h;
      double f2 = a2->g + a2->h;
      if( ( f1 >= f2 - 0.000001) && (f1 <= f2 +0.000001) )
        return a1->g < a2->g; // if equal compare gvals
      return f1 > f2;
    }
  };
  
  struct AState; // forward declaration
  using PriorityQueue = boost::heap::d_ary_heap<AState*, boost::heap::mutable_<true>,
                        boost::heap::arity<2>, boost::heap::compare< compareAStates<AState> >>;
  
  struct AState
  {
    int x,y,z;
    double g = infCost;
    double h;
    bool cl = false;
    AState* parent = nullptr;
    PriorityQueue::handle_type heapkey;
    AState( int x, int y ): x(x), y(y) {}
    AState( int x, int y, int z): x(x), y(y), z(z) {}
  };
  
  struct HashMap
  {
    HashMap(size_t sz)
    {
      if(sz < 1000000 ) hashMapV_.resize(sz);
      else useVec_ = false;
    }
    
    ~HashMap()
    {
      if(useVec_)
        for(auto it:hashMapV_)
          if(it)
            delete it;
      else
        for( auto it = hashMapM_.begin(); it != hashMapM_.end(); ++it )
          if(it->second)
            delete it->second;
    }
        
    AState*& operator[] (size_t n)
    {
      if(useVec_) return hashMapV_[n];
      else return hashMapM_[n];
    }
  private:
    std::vector<AState*> hashMapV_;
    std::unordered_map<size_t, AState*> hashMapM_;
    bool useVec_ = true;
  };
  

  typedef pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> pyArrayXi;
  
  pybind11::tuple planOn2DGrid( pyArrayXi cMap,
                                pyArrayXi start,
                                pyArrayXi goal,
                                double epsilon = 1.0 )
  {
    // Get references to the data and dimensions
    auto cMapData = cMap.unchecked<2>();
    auto startData = start.unchecked<1>();
    auto goalData = goal.unchecked<1>();
    size_t xDim = cMapData.shape(0);
    size_t yDim = cMapData.shape(1);
    size_t cMapLength = xDim*yDim;
    
    // Initialize the HashMap and Heap
    HashMap hm(cMapLength);
    PriorityQueue pq;

    // Initialize the start
    AState *currNode_pt = new AState(startData(0),startData(1));
    currNode_pt->g = 0.0;
    currNode_pt->h = epsilon*std::sqrt( (goalData(0)-startData(0))*(goalData(0)-startData(0)) +
                                        (goalData(1)-startData(1))*(goalData(1)-startData(1)) );
    currNode_pt->cl = true;
    size_t indStart = startData(0) + xDim*startData(1);
    hm[indStart] = currNode_pt;
    
    while(true)
    {
      if( currNode_pt->x == goalData(0) && currNode_pt->y == goalData(1) ) 
        break;
      
      //iterate over neighbor nodes
      for (int xShift=-1; xShift <= 1; ++xShift)
      {
        int xNeighbor = currNode_pt->x + xShift;
        if (xNeighbor < 0) continue; // skip outside of map
        if (xNeighbor >= xDim) continue;
        
        for (int yShift=-1; yShift <= 1; yShift++)
        {
          // skip current node
          if (xShift==0 && yShift==0) continue;
          int yNeighbor = currNode_pt->y + yShift;
          if (yNeighbor < 0) continue;
          if (yNeighbor >= yDim) continue;
         
          if( cMapData(xNeighbor,yNeighbor) > 0 ) continue; // skip collisions

          // initialize if never seen before
          size_t indNeighbor = xNeighbor + xDim*yNeighbor;
          AState*& child_pt = hm[indNeighbor];
          if( !child_pt )
            child_pt = new AState(xNeighbor,yNeighbor);
          
          // skip closed nodes
          if( child_pt->cl ) continue;
          
          // Calculate cost
          double stageCost = (std::abs(xShift) + std::abs(yShift) > 1) ? sqrtTwoCost : 1.0;
          double costNeighbor = currNode_pt->g + stageCost;
          
          if (costNeighbor < child_pt->g)
          {
            //update the heuristic value
            if( !std::isinf(child_pt->g) )
            {
              // node seen before
              child_pt->g = costNeighbor;
              // increase == decrease with our comparator (!)
              pq.increase(child_pt->heapkey);
            }else
            {
              // ADD
              child_pt->h = epsilon*std::sqrt((xNeighbor-goalData(0))*(xNeighbor-goalData(0)) + 
        			                                (yNeighbor-goalData(1))*(yNeighbor-goalData(1)));
        			child_pt->g = costNeighbor;
        			child_pt->heapkey = pq.push(child_pt);       			
            }
            child_pt->parent = currNode_pt;
          }
        }
      }
      if( pq.empty() )
        return pybind11::make_tuple(infCost,std::list<std::array<int,2>>());
      currNode_pt = pq.top(); pq.pop(); // get element with smallest cost
      currNode_pt->cl = true;
    }

    
    // Planning done; Recover the path
    double pcost = currNode_pt->g;
    std::list<std::array<int,2>> path;
    while( currNode_pt->parent )
    {
      path.push_front({currNode_pt->x,currNode_pt->y});
      currNode_pt = currNode_pt->parent;
    }
    path.push_front({currNode_pt->x,currNode_pt->y});
    
    return pybind11::make_tuple(pcost,path);
  }
  
  
  
  
  pybind11::tuple planOn3DGrid( pyArrayXi cMap,
                                pyArrayXi start,
                                pyArrayXi goal,
                                double epsilon = 1.0 )
  {
    // Get references to the data and dimensions
    auto cMapData = cMap.unchecked<3>();
    auto startData = start.unchecked<1>();
    auto goalData = goal.unchecked<1>();
    size_t xDim = cMapData.shape(0);
    size_t yDim = cMapData.shape(1);
    size_t zDim = cMapData.shape(2);
    size_t xyDim = xDim*yDim;
    size_t cMapLength = xyDim*zDim;
    
    // Initialize the HashMap and Heap
    HashMap hm(cMapLength);
    PriorityQueue pq;
    
    // Initialize the start
    AState *currNode_pt = new AState(startData(0),startData(1),startData(2));
    currNode_pt->g = 0.0;
    currNode_pt->h = epsilon*std::sqrt( (goalData(0)-startData(0))*(goalData(0)-startData(0)) +
                                        (goalData(1)-startData(1))*(goalData(1)-startData(1)) +
                                        (goalData(2)-startData(2))*(goalData(2)-startData(2)) );
    currNode_pt->cl = true;
    size_t indStart = startData(0) + xDim*startData(1) + xyDim*startData(2);// colmajor
    hm[indStart] = currNode_pt;

    while(true)
    {
      if( currNode_pt->x == goalData(0) && currNode_pt->y == goalData(1) && currNode_pt->z == goalData(2) ) 
        break;

      //iterate over neighbor nodes
      for (int xShift=-1; xShift <= 1; ++xShift)
      {
        int xNeighbor = currNode_pt->x + xShift;
        if (xNeighbor < 0) continue; // skip outside of map
        if (xNeighbor >= xDim) continue;
        
        for (int yShift=-1; yShift <= 1; yShift++)
        {
          int yNeighbor = currNode_pt->y + yShift;
          if (yNeighbor < 0) continue;
          if (yNeighbor >= yDim) continue;  
          
          for (int zShift=-1; zShift <= 1; zShift++)
          {
            if (xShift==0 && yShift==0 && zShift==0) continue; // skip current node
            int zNeighbor = currNode_pt->z + zShift;
            if (zNeighbor < 0) continue;
            if (zNeighbor >= zDim) continue;          

            // skip collisions
            if( cMapData(xNeighbor,yNeighbor,zNeighbor) > 0 ) continue;
            

            // initialize if never seen before
            size_t indNeighbor = xNeighbor + xDim*yNeighbor + xyDim*zNeighbor;
            AState*& child_pt = hm[indNeighbor];
            if( !child_pt )
              child_pt = new AState(xNeighbor,yNeighbor,zNeighbor);
            
            if( child_pt->cl ) continue; // skip closed nodes         
            
                        
            // Calculate cost
            double stageCost = 1.0; // get the cost multiplier
            switch (std::abs(xShift) + std::abs(yShift)+ std::abs(zShift)){
              case 2: stageCost = sqrtTwoCost; break;
              case 3: stageCost = sqrtThreeCost; break;
            }
            double costNeighbor = currNode_pt->g + stageCost;

            if (costNeighbor < child_pt->g)
            {
              //update the heuristic value
              if( !std::isinf(child_pt->g) )
              {
                // node seen before
                child_pt->g = costNeighbor;
                // increase == decrease with our comparator (!)
                pq.increase(child_pt->heapkey);
              }else
              {
                // ADD
                child_pt->h = epsilon*std::sqrt((xNeighbor-goalData(0))*(xNeighbor-goalData(0)) + 
          			                                (yNeighbor-goalData(1))*(yNeighbor-goalData(1)) +
          			                                (zNeighbor-goalData(2))*(zNeighbor-goalData(2)));
          			child_pt->g = costNeighbor;
          			child_pt->heapkey = pq.push(child_pt);
              }
              child_pt->parent = currNode_pt;
            }      
          }
        }      
      } //END: iterate over neighbor nodes
      
      if( pq.empty() )
        return pybind11::make_tuple(infCost,std::list<std::array<int,3>>());
      currNode_pt = pq.top(); pq.pop(); // get element with smallest cost
      currNode_pt->cl = true;      
    } // END: while 
    
    // Planning done; Recover the path
    double pcost = currNode_pt->g;
    std::list<std::array<int,3>> path;
    while( currNode_pt->parent )
    {
      path.push_front({currNode_pt->x,currNode_pt->y,currNode_pt->z});
      currNode_pt = currNode_pt->parent;
    }
    path.push_front({currNode_pt->x,currNode_pt->y,currNode_pt->z});
    
    return pybind11::make_tuple(pcost,path);    
  } 
}



PYBIND11_MODULE(astar_py, m)
{
  m.doc() = "Pybind11 A* plugin";

  m.def("planOn2DGrid", &astar::planOn2DGrid,
    "Plan a path on a 2D grid using A*.\n\n"
    "Input:\n"
    "\tcMap: nxm matrix\n"
    "\tstart: 2d vector\n"
    "\tgoal: 2d vector\n"
    "\tepsilon: heuristic weighting\n",
    pybind11::arg("cMap"),
    pybind11::arg("start"),
    pybind11::arg("goal"),
    pybind11::arg("epsilon") = 1.0);

  m.def("planOn3DGrid", &astar::planOn2DGrid,
    "Plan a path on a 3D grid using A*.\n\n"
    "Input:\n"
    "\tcMap: n x m x p matrix\n"
    "\tstart: 3d vector\n"
    "\tgoal: 3d vector\n"
    "\tepsilon: heuristic weighting\n",
    pybind11::arg("cMap"),
    pybind11::arg("start"),
    pybind11::arg("goal"),
    pybind11::arg("epsilon") = 1.0);
  
  m.attr("__version__") = "dev";
}


